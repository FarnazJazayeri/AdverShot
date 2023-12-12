# coding=utf-8
import torch
from numpy import dtype
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from learner import Learner
from attack import PGD


class PrototypicalLoss(nn.Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


# def prototypical_loss(input, target, n_support):
def prototypical_loss(out_spt, y_spt, out_qry, y_qry):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    # target_cpu = target.to('cpu')
    # input_cpu = input.to('cpu')
    out_spt_cpu, y_spt_cpu, out_qry_cpu, y_qry_cpu = out_spt.to('cpu'), y_spt.to('cpu'), out_qry.to('cpu'), y_qry.to(
        'cpu')

    n_classes = len(set(y_spt.tolist()))
    n_query = out_qry.size()[0]

    # def supp_idxs(c):
    # FIXME when torch will support where as np
    #    return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    # classes = torch.unique(target_cpu)
    # n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    # n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    # support_idxs = list(map(supp_idxs, classes))

    # prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    prototypes = []
    for c in range(n_classes):
        sample_class_i_idx = torch.IntTensor([i for i, v in enumerate(y_spt_cpu.tolist()) if v == c])
        sample_class_i = out_spt_cpu.index_select(dim=0, index=sample_class_i_idx)
        prototypes.append(sample_class_i.mean(dim=0))

    prototypes = torch.stack(prototypes, dim=0)

    # FIXME when torch will support where as np
    # query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    # query_samples = input.to('cpu')[query_idxs]

    # dists = euclidean_dist(query_samples, prototypes)
    dists = euclidean_dist(out_qry_cpu, prototypes)
    criterion = nn.CrossEntropyLoss().to('cpu')
    log_p_y = F.log_softmax(-dists, dim=1).view(n_query, n_classes)
    loss_val = criterion(log_p_y, y_qry_cpu)
    y_hat = torch.argmax(log_p_y, 1)

    acc_val = y_hat.eq(y_qry_cpu).float().mean()

    return loss_val, acc_val

##################
class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.adv_eps = args.adv_attack_eps
        self.adv_alpha = args.adv_attack_alpha
        self.adv_iters = args.adv_attack_iters
        self.weight_robust = args.weight_robust

        self.net = Learner(config, args.imgc, args.imgsz)
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

    def forward(self, x_spt, y_spt, x_qry, y_qry, need_adv=True):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        loss_q = 0.0
        correct_q = 0.0
        loss_q_adv = 0.0
        correct_q_adv = 0.0
        eps, step = (self.adv_eps, self.adv_iters)
        at = PGD(eps=eps, sigma=self.adv_alpha, nb_iter=step)  ####################
        for i in range(task_num):
            self.meta_optim.zero_grad()
            # x, y = batch
            # x, y = x.to(device), y.to(device)

            out_spt = self.net(x_spt[i], self.net.parameters(), bn_training=True)
            out_qry = self.net(x_qry[i], self.net.parameters(), bn_training=True)
            # loss, acc = loss_fn(model_output, target=y,
            #                    n_support=opt.num_support_tr)
            loss, acc = prototypical_loss(out_spt, y_spt[i], out_qry, y_qry[i])
            #################
            out_qry_adv = at.attack(self.net, self.net.parameters(), x_qry[i], y_qry[i],
                                    extra_params=dict(x_spt=x_spt[i], y_spt=y_spt[i]))
            out_qry_adv = self.net(out_qry_adv, self.net.parameters(), bn_training=True)
            loss_adv, acc_adv = prototypical_loss(out_spt, y_spt[i], out_qry_adv, y_qry[i])
            ###################
            loss_q += loss
            correct_q += acc
            ##############
            loss_q_adv += loss_adv
            correct_q_adv += acc_adv
            ##############
            # loss.backward()
            loss_total = loss + self.weight_robust*loss_adv
            loss_total.backward()
            self.meta_optim.step()
            # train_loss.append(loss.item())
            # train_acc.append(acc.item())
        loss_q /= task_num
        correct_q /= task_num
        loss_q_adv /= task_num
        correct_q_adv /= task_num

        return correct_q, loss_q, correct_q_adv, loss_q_adv

    def test(self, x_spt, y_spt, x_qry, y_qry, need_adv=False):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        out_spt = self.net(x_spt, self.net.parameters(), bn_training=True)
        out_qry = self.net(x_qry, self.net.parameters(), bn_training=True)
        loss, acc = prototypical_loss(out_spt, y_spt, out_qry, y_qry)
        ####### robust ##########
        eps, step = (self.adv_eps, self.adv_iters)
        at = PGD(eps=eps, sigma=self.adv_alpha, nb_iter=step)  ####################
        out_qry_adv = at.attack(self.net, self.net.parameters(), x_qry, y_qry,
                                extra_params=dict(x_spt=x_spt, y_spt=y_spt))
        out_qry_adv = self.net(out_qry_adv, self.net.parameters(), bn_training=True)
        loss_adv, acc_adv = prototypical_loss(out_spt, y_spt, out_qry_adv, y_qry)
        ###################

        return acc, loss, acc_adv, loss_adv
