import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

class Learner(nn.Module):
    def __init__(self, model_name):
        super(Learner, self).__init__()
        self.model_name = model_name
        if model_name == "resnet_18":
            self.model = models.resnet18(pretrained=False)
        self.extract_vars_bn()

    def extract_vars_bn(self):
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.vars_bn.extend(module.parameters())
            elif hasattr(module, 'weight'):
                # Append the weights of the module to the list
                self.vars.append(module.weight)
        for param in self.vars_bn:
            param.requires_grad = False

    def assign_vars_bn(self, vars, vars_bn):
        param_index = 0
        vars_iter = iter(vars)
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # Assign the modified parameters to the batch normalization layer
                module.running_mean = vars_bn[param_index]
                module.running_var = vars_bn[param_index + 1]
                param_index += 2
            elif hasattr(module, 'weight'):
                module.weight.data = next(vars_iter)

    def forward(self):

### testing
learner = Learner(model_name='resnet_18')
vars = learner.vars
vars_bn = learner.vars_bn

learner.assign_vars_bn(vars, vars_bn)