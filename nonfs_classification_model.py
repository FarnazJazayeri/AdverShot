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

    def __init__(self, args, config, at=False):
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
        self.meta_lr = args.meta_lr
        #self.weight = args.weight
        #if self.weight is not None:
        #    params =
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.at = at

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
        self.meta_optim.zero_grad()
        need_adv = self.at
        b, c, h, w = x.size()
        eps, step = (self.adv_eps, self.adv_iters)
        at = PGD(eps=eps, sigma=self.adv_alpha, nb_iter=step)  ####################
        ###############################
        self.meta_optim.zero_grad()
        out = self.net(x, self.net.parameters(), bn_training=True)
        #print(out.shape, y.shape)
        loss = self.criterion(out, y)
        pred = F.softmax(out, dim=1).argmax(dim=1)
        acc = np.array(torch.eq(pred, y).sum().item())
        #################
        if need_adv:
            x_adv = at.attack(self.net, self.net.parameters(), x, y)
            out_adv = self.net(x_adv, self.net.parameters(), bn_training=True)
            loss_adv = self.criterion(out_adv, y)
            #acc_adv = predict(out_adv, y)
            pred_adv = F.softmax(out_adv, dim=1).argmax(dim=1)
            acc = np.array(torch.eq(pred_adv, y).sum().item())
            ###
            loss_total = loss + loss_adv
            loss_total.backward()
        else:
            loss_adv = torch.Tensor([0.0]).to(x.device)
            acc_adv = 0.0
            loss.backward()
        self.meta_optim.step()

        return acc, loss, acc_adv, loss_adv

    def test(self, x, y, need_adv=False):
        need_adv = self.at
        b, c, h, w = x.size()
        eps, step = (self.adv_eps, self.adv_iters)
        at = PGD(eps=eps, sigma=self.adv_alpha, nb_iter=step)  ####################
        ###############################
        self.meta_optim.zero_grad()
        out = self.net(x, self.net.parameters(), bn_training=True)
        loss = self.criterion(out, y)
        pred = F.softmax(out, dim=1).argmax(dim=1)
        acc = np.array(torch.eq(pred, y).sum().item())
        #################
        if need_adv:
            x_adv = at.attack(self.net, self.net.parameters(), x, y)
            out_adv = self.net(x_adv, self.net.parameters(), bn_training=True)
            loss_adv = self.criterion(out_adv, y)
            pred_adv = F.softmax(out_adv, dim=1).argmax(dim=1)
            acc = np.array(torch.eq(pred_adv, y).sum().item())
            ###
            loss_total = loss + loss_adv
            #loss_total.backward()
        else:
            loss_adv = 0.0
            acc_adv = 0.0
            #loss.backward()
        return acc, loss, acc_adv, loss_adv