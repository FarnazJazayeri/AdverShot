import torch
import torch.nn as nn

from tools.attacks.attack import Attack
from source.prototypical.src.prototypical_loss import prototypical_loss as loss_fn


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, device=None, eps=8/255, loss_type="CrossEntropyLoss", num_support=5):
        super().__init__('FGSM', model, device)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']
        self.loss_type = loss_type
        self.num_support = num_support

    def perturb(self, images, labels):
        r"""
        Overridden.
        """


        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if self.loss_type=="CrossEntropyLoss":
            loss = nn.CrossEntropyLoss()
        if self.loss_type=="PrototypicalLoss":
            loss = loss_fn 

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            if self.loss_type=="PrototypicalLoss":
                cost, acc = -loss(outputs, target_labels, self.num_support)
            else:
                cost = -loss(outputs, target_labels)
        else:
            if self.loss_type=="PrototypicalLoss":
                cost, acc = loss(outputs, labels, self.num_support)
            else:
                cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images