import torch
from torch import nn
from torch.nn import functional as F

from torch import optim
import numpy as np

from copy import deepcopy
from collections import OrderedDict

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, learner, update_lr, meta_lr, adv_reg_param, update_steps, update_steps_test, attacker=None):
        """
        :param learner: The few-shot learner model
        :param update_lr: Learning rate for few-shot learner
        :param meta_lr: Learning rate for meta learner
        :param update_steps: number of SGD iterations for few-shot learner during training
        :param update_steps_test: number of SGD iterations for few-shot learner during testing

        """
        super(Meta, self).__init__()
        self.learner = learner
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.adv_reg_param = adv_reg_param
        self.update_step = update_steps
        self.update_step_test = update_steps_test
        self.attacker = attacker
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.meta_lr)

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

    def fit(self, train_dl, validation_dl, num_epochs=100):
        """
        Meta-train procedure.
        :param dataloader: The dataloader to generate batches
        :param num_epochs: number of epochs to train
        :return:
        """
        for epoch in range(num_epochs):
            for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_dl):
                meta_loss, losses_q, losses_q_adv, accs, accs_adv = self.forward(x_spt, y_spt, x_qry, y_qry)
                self.meta_optim.zero_grad()
                meta_loss.backward()
                self.meta_optim.step()

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   list of [batch_size, num_channels, height, width] (len=spt_size)
        :param y_spt:   list of [batch_size] (len=spt_size)
        :param x_qry:   list of [batch_size, num_channels, height, width] (len=qry_size)
        :param y_qry:   list of [batch_size] (len=qry_size)

        For each task, trains the learner on support set and calculates cross-entropy loss on both query set and adversarially attacked query set.
        Averages these losses and adversarial losses for different tasks to obtain average loss and average adversarial loss.

        calculates the meta-loss by combining average loss and average adversarial loss: avg_loss + (lambda * avg_adv_loss)
        :return: meta-loss
        """

        x_spt = torch.stack(x_spt, dim=1)
        y_spt = torch.stack(y_spt, dim=1)
        x_qry = torch.stack(x_qry, dim=1)
        y_qry = torch.stack(y_qry, dim=1)

        batch_size, spt_size, c_, h, w = x_spt.size()
        qry_size = x_qry.size()[1]

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        losses_q_adv = [0 for _ in range(self.update_step + 1)]
        corrects_adv = [0 for _ in range(self.update_step + 1)]

        optimizer = torch.optim.SGD(self.learner.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)

        for i in range(batch_size):
            fast_weights = OrderedDict(self.learner.named_parameters())
            # fast_weights = deepcopy(list(self.learner.named_parameters()))
            optimizer.zero_grad()
            loss_q, correct, loss_q_adv, correct_adv = self.evaluate_learner(
                fast_weights=fast_weights,
                data=x_qry[i],
                label=y_qry[i],
            )
            optimizer.zero_grad()
            losses_q[0] += loss_q
            corrects[0] = corrects[0] + correct
            losses_q_adv[0] += loss_q_adv
            corrects_adv[0] = corrects_adv[0] + correct_adv

            # TODO How to implement the meta-train procedure for MAML and Prototypical
            for k in range(1, self.update_step):
                logits = self.learner(x_spt[i], fast_weights)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, list(fast_weights.values()))
                fast_weights = OrderedDict(map(lambda p: (p[1][0], p[1][1] - self.update_lr * p[0]), zip(grad, fast_weights.items())))

                optimizer.zero_grad()
                loss_q, correct, loss_q_adv, correct_adv = self.evaluate_learner(
                    fast_weights=fast_weights,
                    data=x_qry[i],
                    label=y_qry[i],
                )
                optimizer.zero_grad()
                losses_q[k] += loss_q
                corrects[k] += correct
                losses_q_adv[k] += loss_q_adv
                corrects_adv[k] += correct_adv

        loss_q = losses_q[-1] / batch_size
        accs = np.array(corrects) / (qry_size * batch_size)
        loss_q_adv = losses_q_adv[-1] / batch_size
        accs_adv = np.array(corrects_adv) / (qry_size * batch_size)

        final_loss = loss_q + self.adv_reg_param * loss_q_adv
        return final_loss, losses_q, losses_q_adv, accs, accs_adv

    def evaluate_learner(self, fast_weights, data, label):
        """
                :param fast_weights: model parameters
                :param data:  data to evaluate on
                :param label: true labels of data


                :return: (
                    loss: cross-entropy loss on data,
                    correct: number of correct predictions,
                    loss_adv: cross-entropy loss on adversarially perturbed data,
                    correct_adv: number of correct predictions on adversarially perturbed data
                )
                """

        adv_data = self.attacker.attack(self.learner, fast_weights, data, label)

        with torch.no_grad():
            logits_q = self.learner(data, fast_weights)
            loss = F.cross_entropy(logits_q, label)

            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, label).sum().item()

            logits_q_adv = self.learner(adv_data, fast_weights)
            loss_adv = F.cross_entropy(logits_q_adv, label)

            pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
            correct_adv = torch.eq(pred_q_adv, label).sum().item()

        return loss, correct, loss_adv, correct_adv
