import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from torch import optim
import numpy as np

from copy import deepcopy
from attack import PGD


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, learner):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = learner
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def fit(self, dataloader, num_epochs=100):
        for epoch in range(num_epochs):
            for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
                meta_loss, losses_q, losses_q_adv, accs, accs_adv = self.forward(x_spt, y_spt, x_qry, y_qry)
                self.meta_optim.zero_grad()
                meta_loss.backward()
                self.meta_optim.step()


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        eps, step = (4.0, 10)
        losses_q_adv = [0 for _ in range(self.update_step + 1)]
        corrects_adv = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))



            at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
            data = x_qry[i]
            label = y_qry[i]
            optimizer.zero_grad()
            adv_inp_adv = at.attack(self.net, fast_weights, data, label)
            optimizer.zero_grad()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct


            data = x_qry[i]
            label = y_qry[i]
            optimizer.zero_grad()
            adv_inp = at.attack(self.net, self.net.parameters(), data, label)
            optimizer.zero_grad()
            with torch.no_grad():
                logits_q_adv = self.net(adv_inp, self.net.parameters(), bn_training=True)
                loss_q_adv = F.cross_entropy(logits_q_adv, label)
                losses_q_adv[0] += loss_q_adv

                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, label).sum().item()
                corrects_adv[0] = corrects_adv[0] + correct_adv

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

                logits_q_adv = self.net(adv_inp_adv, fast_weights, bn_training=True)
                loss_q_adv = F.cross_entropy(logits_q_adv, label)
                losses_q_adv[1] += loss_q_adv

                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, label).sum().item()
                corrects_adv[1] = corrects_adv[1] + correct_adv

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
                data = x_qry[i]
                label = y_qry[i]
                optimizer.zero_grad()
                adv_inp_adv = at.attack(self.net, fast_weights, data, label)
                optimizer.zero_grad()

                logits_q_adv = self.net(adv_inp_adv, fast_weights, bn_training=True)
                loss_q_adv = F.cross_entropy(logits_q_adv, label)
                losses_q_adv[k + 1] += loss_q_adv

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

                    pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                    correct_adv = torch.eq(pred_q_adv, label).sum().item()
                    corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        loss_q_adv = losses_q_adv[-1] / task_num

        final_loss = loss_q + self.adv_reg_param * loss_q_adv

        accs = np.array(corrects) / (querysz * task_num)
        accs_adv = np.array(corrects_adv) / (querysz * task_num)

        return final_loss, losses_q, losses_q_adv, accs, accs_adv