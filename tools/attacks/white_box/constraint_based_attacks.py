import torch
import torch.nn.functional as F


class LinfPGDAttack(object):
    #eps = 0.0314
    #k = 7
    def __init__(self, model, criterion, eps, k, alpha):
        self.model = model
        self.criterion = criterion
        self.eps = eps
        self.k = k
        self.alpha = alpha
        
    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                #loss = F.cross_entropy(logits, y)
                loss = self.criterion(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.eps), x_natural + self.eps)
            x = torch.clamp(x, 0, 1)
        return x