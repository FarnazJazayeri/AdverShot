import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class MAMLNet(nn.Module):
    '''
    MAML Model
    '''
    def __init__(self, n_way, x_dim=1, hid_dim=64, z_dim=32):
        super(MAMLNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        # Add fully connected layers after the feature extraction
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim * 4 * 4, 256),  # Adjust the input size based on your feature map size
            nn.ReLU(),
            nn.Linear(256, n_way),  # Output layer for classification
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_layers(x)
        return x
