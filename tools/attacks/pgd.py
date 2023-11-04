import torch
import torch.nn as nn
import numpy as np
from tools.attacks.attack import Attack
from tools.attacks.utils import class_tensor_extract
import torch.nn.functional as F
from tools.losses.prototypical_loss import prototypical_loss as loss_fn


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=8/255, alpha=4/255, steps=10, random_start=True,
                 tuning_method=None, num_population=0, num_features=0, num_classes=11, con_features=[[0.01, 1], [0.01, 0.1]], mask=False, mask_size=8, loss_type="CrossEntropyLoss", num_support=5):
        super().__init__('PGD', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.tuning_method = tuning_method
        self.num_population = num_population
        self.num_features = num_features
        self.num_classes = num_classes
        self.con_features = con_features
        self.val = 0.0
        self.update_by_val_index = 0
        self.w_f1 = 1.0
        self.w_f2 = 2.0
        self.mask = mask
        self.mask_size = mask_size
        self.loss_type = loss_type
        self.num_support = num_support
        
    def init_population(self, mask):
        if mask:
            p = []
            population = np.random.uniform(0, 1, size = (self.num_population, self.mask_size, self.mask_size)) # num_population x mask_size x mask_size
        else:
            p = []
            for i in range(self.num_features):
                p.append(np.random.uniform(self.con_features[i][0], self.con_features[i][1], size=(self.num_population, self.num_classes)))
            population = np.transpose(np.stack(p), (1, 0, 2))  # num_population x num_features x num_classes
        return population
    
    def calc_fitness_function(self, cost, adv_images, images, labels, population):
        # population: num_features x num_classes
        # labels: B x 1
        
        # Calc the grad
        #print(population.shape)
        #adv_images.requires_grad = True
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        # Extract features
        indices_tensor_alpha = labels 
        features_alpha = population[1][indices_tensor_alpha]
        indices_tensor_eps = labels
        features_eps = population[0][indices_tensor_eps]
        # Update the adv images
        #print(adv_images.shape, torch.tensor(features_alpha).shape, grad.shape)
        adv_images = adv_images.detach() + torch.tensor(features_alpha).view(2, 1, 1, 1)*grad.sign()
        for i in range(labels.shape[0]):
            delta = torch.clamp(adv_images[i] - images[i], min=-features_eps[i], max=features_eps[i])
            adv_images[i] = torch.clamp(images[i] + delta, min=0, max=1).detach()
        # Calc the cost
        adv_images.requires_grad = True
        outputs = self.get_logits(adv_images.float())
        cost = self.loss(outputs, labels)
        return cost, adv_images

    def update_features(self, val=0.0):
        if val >= self.val:
            self.val = val
        else:
            self.update_by_val_index += 1
        if self.update_by_val_index == 5:
            print("Update adversarial hyperparameters !!!")
            self.update_by_val_index = 0
            self.con_features[0][0] += 8/255
            self.con_features[1][0] += 4/255
            # self.w_f2 = self.w_f2 / 1.1
            self.con_features[0][0] = np.min([self.con_features[0][1] - 8/255, 64/255])
            self.con_features[1][0] = np.min([self.con_features[1][1] - 4/255, 0.2])
    
    def perturb(self, images, labels, new_eps=None, new_alpha=None, tune=False, val=0.0, val_mode=False, model=None):
        r"""
        Overridden.
        """
        #############
        if model is not None:
            model.eval()
            self.model = model
        ###############
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
            
        
        if self.loss_type=="CrossEntropyLoss":
            self.loss = nn.CrossEntropyLoss()
        if self.loss_type=="PrototypicalLoss":
            self.loss = loss_fn
        #self.loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        ###
        if new_eps is not None:
            eps = new_eps
            self.con_features[0][0] = eps - 2/255
            self.con_features[0][1] = eps + 4/255
        else:
            eps = self.eps
        if new_alpha is not None:
            self.con_features[1][0] = new_alpha - 2/255
            self.con_features[1][1] = new_alpha + 2/255
        
        ###
        
        if self.random_start:
            # Starting at a uniformly random point
            if isinstance(eps, (int, float)):
                adv_images = adv_images + \
                    torch.empty_like(adv_images).uniform_(-eps, eps) ########################
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            elif isinstance(eps, list):
                adv_images = adv_images + \
                    torch.empty_like(adv_images).uniform_(-eps[0], eps[0]) ########################
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            else:
                print("The value is neither a floating-point number or a list!!!")
            #print(torch.empty_like(adv_images).uniform_(-eps, eps).max(), torch.empty_like(adv_images).uniform_(-eps, eps).min())
            
        
        if val_mode:
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images) ###################### model prediction #################
    
                # Calculate loss
                if self.targeted:
                    cost = -self.loss(outputs, target_labels)
                else:
                    # print(type(outputs), type(labels))
                    cost = self.loss(outputs, labels) ########################## loss value #########################
                #print(f"cost: {cost}")
                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
                adv_images = adv_images.detach() + self.alpha*grad.sign()
                if isinstance(eps, (int, float)):
                    delta = torch.clamp(adv_images - images,
                                        min=-eps, max=eps)
                    adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        else:
            if self.tuning_method is None or not tune and not self.val:
                for _ in range(self.steps):
                    adv_images.requires_grad = True
                    outputs = self.get_logits(adv_images) ###################### model prediction #################
        
                    # Calculate loss
                    if self.targeted:
                        cost = -self.loss(outputs, target_labels)
                    else:
                        if self.loss_type=="PrototypicalLoss":
                        # print(type(outputs), type(labels))
                            cost, acc = self.loss(outputs, labels, self.num_support) ########################## loss value #########################
                        else:
                            cost = self.loss(outputs, labels)
                    #print(f"cost: {cost}")
                    # Update adversarial images
                    if self.mask: #################### mask ################
                        grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]
                        #mask = torch.rand((grad.size(0), 1, self.mask_size, self.mask_size))
                        mask = torch.rand((1, 1, self.mask_size, self.mask_size))
                        mask = F.interpolate(mask, size=(grad.size(2), grad.size(3)), mode='bilinear', align_corners=True)
                        mask = mask.to(adv_images.device)
                        adv_images = adv_images.detach() + self.alpha*mask*grad.sign()
                            
                    else:
                        grad = torch.autograd.grad(cost, adv_images,
                                               retain_graph=False, create_graph=False)[0]
                        adv_images = adv_images.detach() + self.alpha*grad.sign()
                    if isinstance(eps, (int, float)):
                        delta = torch.clamp(adv_images - images,
                                            min=-eps, max=eps)
                        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
        
        
class PGDL2(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=1.0, alpha=0.2, steps=10, random_start=True, eps_for_division=1e-10):
        super().__init__('PGDL2', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ['default', 'targeted']

    def perturb(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            print("grad", grad.min(), grad.max())
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division  # nopep8
            print("grad_norms", grad_norms.min(), grad_norms.max())
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            print("delta", delta.min(), delta.max())
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            print("delta_norm", delta_norms.min(), delta_norms.max())
            factor = self.eps / delta_norms
            print("factor 1", factor.min(), factor.max())
            factor = torch.min(factor, torch.ones_like(delta_norms))
            print("factor 2", factor.min(), factor.max())
            delta = delta * factor.view(-1, 1, 1, 1)
            print("delta new", torch.norm(delta, p=2))
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
