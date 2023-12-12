# coding=utf-8
import torch
from numpy import dtype
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from learner import Learner
from attack import PGD


def predict(outputs, labels, threshold=None):
    predictions = []
    _, predicted_classes = torch.max(outputs, dim=1)
    predicted_probabilities, _ = torch.max(outputs, dim=1)
    predictions.append(predicted_classes.cpu().numpy())
    predictions = np.concatenate(predictions)
    ###
    label_list = []
    label_np = labels.cpu().numpy()
    result = (label_np == predictions).astype(int)
    result = result.sum()
    return result


class NonFSModel(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(NonFSModel, self).__init__()

        self.update_lr = args.update_lr
        self.n_way = args.n_way
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.adv_eps = args.adv_attack_eps
        self.adv_alpha = args.adv_attack_alpha
        self.adv_iters = args.adv_attack_iters

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.update_lr)
        self.criterion = nn.CrossEntropyLoss()

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

    def forward(self, x, y, need_adv=True):
        b, c, h, w = x.size()
        eps, step = (self.adv_eps, self.adv_iters)
        at = PGD(eps=eps, sigma=self.adv_alpha, nb_iter=step)  ####################
        ###############################
        self.meta_optim.zero_grad()
        out = self.net(x, self.net.parameters(), bn_training=True)
        loss = self.criterion(out, y)

        _, predicted = torch.max(out, 1)
        acc = (predicted == y).sum().item()
        # acc = predict(out, y)

        #################
        if need_adv:
            y_adv = at.attack(self.net, self.net.parameters(), x, y)
            out_adv = self.net(y_adv, self.net.parameters(), bn_training=True)
            loss_adv = self.criterion(out, y)

            _, predicted = torch.max(out_adv, 1)
            acc_adv = (predicted == y).sum().item()
            # acc_adv = predict(out_adv, y)

            ###
            # loss_total = loss + loss_adv
            # loss_total.backward()
        else:
            loss_adv = 0.0
            acc_adv = 0.0
            # loss.backward()

        return acc, loss, acc_adv, loss_adv

    def test(self, x, y, need_adv=False):

        eps, step = (self.adv_eps, self.adv_iters)
        at = PGD(eps=eps, sigma=self.adv_alpha, nb_iter=step)
        ###############################
        self.meta_optim.zero_grad()
        out = self.net(x, self.net.parameters(), bn_training=True)
        loss = self.criterion(out, y)
        acc = predict(out, y)
        #################
        if need_adv:
            x_adv = at.attack(self.net, self.net.parameters(), x, y)
            out_adv = self.net(x_adv, self.net.parameters(), bn_training=True)
            loss_adv = self.criterion(out_adv, y)
            acc_adv = predict(out_adv, y)
            ###
            loss_total = loss + loss_adv
            loss_total.backward()
        else:
            loss_adv = 0.0
            acc_adv = 0.0
            loss.backward()
        return acc, loss, acc_adv, loss_adv