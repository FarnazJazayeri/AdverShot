import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


### Customized blocks
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))


def set_batchnorm_training_mode(network, train_mode=True):
    for module in network.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.train(train_mode)


### Main networks
class ModelClassification(nn.Module):
    def __init__(self, model_name=None, in_channel=1, hidden_channel=64, emb_channel=64, num_classes=11, weight=None):
        super().__init__()
        self.model_name = model_name

        ## Checkpoint 2
        if model_name == "generic_protonet":
            self.blackbone = nn.Sequential(
                conv_block(in_channel, hidden_channel),
                conv_block(hidden_channel, hidden_channel),
                conv_block(hidden_channel, hidden_channel),
                conv_block(hidden_channel, emb_channel)
            )

        ## Checkpoint 2
        elif model_name == "generic_metanet":
            # this dict contains all tensors needed to be optimized
            self.vars = nn.ParameterList()
            # running_mean and running_var
            self.vars_bn = nn.ParameterList()
            self.blackbone = nn.Sequential(
                conv_block(in_channel, hidden_channel),
                conv_block(hidden_channel, hidden_channel),
                conv_block(hidden_channel, hidden_channel),
                conv_block(hidden_channel, emb_channel)
            )

        ## After checkpoint 2
        elif model_name == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(pretrained=True)
            head = self.backbone.fc
            new_head = nn.Sequential(
                nn.Linear(head.in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            self.backbone.fc = new_head
            if weight:
                self.backbone.load_state_dict(torch.load(weight))
            for params in self.backbone.parameters():
                params.requires_grad = True
            for params in self.backbone.fc.parameters():
                params.requires_grad = True

        elif model_name == 'swin_base':
            self.backbone = models.swin_v2_t(weights=None)
            ### adjust the head
            linear = self.backbone.head
            new_linear = nn.Sequential(
                nn.Linear(linear.in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.backbone.head = new_linear
            ### adjust the feature0
            feature0 = self.backbone.features[0]
            feature0[0] = nn.Conv2d(in_channel, 96, kernel_size=(3, 3))
            self.backbone.features[0] = feature0
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise "This model is not implemented!"

        if weight:
            print(f'Load the model from the checkpoint: {weight}')
            self.load_state_dict(torch.load(weight))

    def forward(self, x, vars=None, bn_training=True):
        if self.model_name == "generic_protonet":
            output = self.backbone(x)
        elif self.model_name == "generic_metanet":
            if vars is None:
                vars = self.vars
            set_batchnorm_training_mode(self.backbone, train_mode=bn_training)
            output = self.backbone(x)

        return output


