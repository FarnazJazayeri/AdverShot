import torch
from torch import nn
from torch.nn import functional as F


# import torchvision.models as models

class Learner(nn.Module):
    def __init__(self, args):
        super(Learner, self).__init__()
        self.model_name = args.model_name
        if args.model_name == "resnet18_maml" or args.model_name == "resnet18_maml_at":
            from model.resnet import ResNet18
            if args.data_name == "omniglot":
                self.model = ResNet18(num_classes=args.n_way, in_channels=1)
            else:
                self.model = ResNet18(num_classes=args.n_way, in_channels=3)
            ### Test
            '''
            self.model = nn.Sequential(
                  nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0, bias=True),
                  nn.ReLU(True),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
                  nn.ReLU(True),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
                  nn.ReLU(True),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
                  nn.ReLU(True),
                  nn.BatchNorm2d(64),
                  nn.Flatten(),
                  nn.Linear(64, 5)
              )
              '''

            for param in self.model.parameters():
                param.requires_grad = True
        ###
        elif args.model_name == "resnet18_protonet" or args.model_name == "resnet18_protonet_at":
            from model.resnet import ResNet18
            if args.data_name == "omniglot":
                self.model = ResNet18(num_classes=args.emb_channel, in_channels=1)
            else:
                self.model = ResNet18(num_classes=args.emb_channel, in_channels=3)
            for param in self.model.parameters():
                param.requires_grad = True

        ###
        self.vars, self.vars_bn = self.extract_vars_bn()

    def extract_vars_bn(self):
        # vars = nn.ParameterList()
        # vars_bn = nn.ParameterList()
        vars = []
        vars_bn = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                params = list(module.parameters())
                # params = [param.detach() for param in params]
                params = [param for param in params]
                vars_bn.extend(params)
            elif hasattr(module, 'weight'):
                # Append the weights of the module to the list
                # params = module.weight.detach()
                params = module.weight
                vars.append(params)

        for param in vars:
            param.requires_grad = True
        for param in vars_bn:
            param.requires_grad = False  # False
        return vars, vars_bn

    def assign_vars_bn(self, vars, vars_bn):
        param_index = 0
        if vars is not None:
            vars_iter = iter(vars)
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                # Assign the modified parameters to the batch normalization layer
                if vars_bn is not None:
                    module.running_mean = vars_bn[param_index]
                    module.running_var = vars_bn[param_index + 1]
                    param_index += 2
            elif hasattr(module, 'weight'):
                if vars is not None:
                    vars = next(vars_iter)
                    module.weight.data = vars
                    module.requires_grad_(True)

    def forward(self, x, vars=None, bn_training=True, fast_parameter_return=False, net_params=False):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(bn_training)
        ###
        if vars is not None:

            if net_params:
                self.assign_vars_bn(vars=self.vars, vars_bn=None)
                out = self.model(x)
                self.vars, self.vars_bn = self.extract_vars_bn()
            else:
                # print(vars, type(vars), len(vars))
                self.assign_vars_bn(vars=vars, vars_bn=None)
                out = self.model(x)
                vars, _ = self.extract_vars_bn()
        else:
            self.assign_vars_bn(vars=self.vars, vars_bn=None)
            out = self.model(x)
            self.vars, self.vars_bn = self.extract_vars_bn()

        ###
        if fast_parameter_return:
            return out, vars
        else:
            return out

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """

        return self.vars
        # return self.model.parameters()

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

### testing
# learner = Learner(model_name='resnet_18')
# vars = learner.vars
# vars_bn = learner.vars_bn

# learner.assign_vars_bn(vars, vars_bn)