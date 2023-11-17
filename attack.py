import torch
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from torch.nn import functional as F
import torch.nn as nn


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


def prototypical_output(out_spt, y_spt, out_qry, y_qry, device='cuda'):
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
    # out_spt_cpu, y_spt_cpu, out_qry_cpu, y_qry_cpu = out_spt.to('cpu'), y_spt.to('cpu'), out_qry.to('cpu'), y_qry.to(
    #     'cpu')

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
        sample_class_i_idx = torch.IntTensor([i for i, v in enumerate(y_spt.tolist()) if v == c]).to(device)
        sample_class_i = out_spt.index_select(dim=0, index=sample_class_i_idx)
        prototypes.append(sample_class_i.mean(dim=0))

    prototypes = torch.stack(prototypes, dim=0)

    # FIXME when torch will support where as np
    # query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    # query_samples = input.to('cpu')[query_idxs]

    # dists = euclidean_dist(query_samples, prototypes)
    dists = euclidean_dist(out_qry, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_query, n_classes)
    return log_p_y


class AttackBase(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, net, inp, label, target=None):
        '''

        :param inp: batched images
        :param target: specify the indexes of target class, None represents untargeted attack
        :return: batched adversaril images
        '''
        pass

    @abstractmethod
    def to(self, device):
        pass


def clip_eta(eta, norm, eps, DEVICE=torch.device('cuda:2')):
    '''
    helper functions to project eta into epsilon norm ball
    :param eta: Perturbation tensor (should be of size(N, C, H, W))
    :param norm: which norm. should be in [1, 2, np.inf]
    :param eps: epsilon, bound of the perturbation
    :return: Projected perturbation
    '''

    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    with torch.no_grad():
        avoid_zero_div = torch.tensor(1e-12).to(DEVICE)
        eps = torch.tensor(eps).to(DEVICE)
        one = torch.tensor(1.0).to(DEVICE)

        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            normalize = torch.norm(eta.reshape(eta.size(0), -1), p=norm, dim=-1, keepdim=False)
            normalize = torch.max(normalize, avoid_zero_div)

            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)

            factor = torch.min(one, eps / normalize)
            eta = eta * factor
    return eta


class PGD(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps=6 / 255.0, sigma=3 / 255.0, nb_iter=20,
                 norm=np.inf, DEVICE=torch.device('cuda'),
                 mean=torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std=torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 random_start=True):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.random_start = random_start

    def single_attack(self, net, para, inp, label, eta, target=None, extra_params=None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''

        adv_inp = inp + eta

        # net.zero_grad()
        if extra_params:
            out_spt = net(extra_params['x_spt'], para)
            out_qry = net(adv_inp, para)
            pred = prototypical_output(
                out_spt=out_spt,
                y_spt=extra_params['y_spt'],
                out_qry=out_qry,
                y_qry=label,
                device=self.DEVICE,
            )
        else:
            pred = net(adv_inp, para)

        loss = self.criterion(pred, label)
        grad_sign = torch.autograd.grad(loss, adv_inp,
                                        only_inputs=True, retain_graph=False)[0].sign()

        adv_inp = adv_inp + grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std + self._mean

        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta / self._std

        #         adv_inp = adv_inp + grad_sign * self.eps
        #         adv_inp = torch.clamp(adv_inp, 0, 1)
        #         eta = adv_inp - inp
        #         eta = clip_eta(eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        return eta

    def attack(self, net, para, inp, label, target=None, extra_params=None):

        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()
        # print(torch.min(torch.min(torch.min(inp[0]))))

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attack(net, para, inp, label, eta, target, extra_params=extra_params)
            # print(i)

        # print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std

        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)
