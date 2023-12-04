import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(kernel_size=(5, 1),
                                           stride=(3, 1),
                                           in_channels=in_channels,
                                           out_channels=out_channels,
                                           padding=2))
        self.leaky_relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        return x


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        convs = []
        in_channels = 1
        self.period = period
        for i in range(4):
            out_channels = 2 ** (5 + i)
            convs.append(ConvBlock(in_channels=in_channels,
                                   out_channels=out_channels))
            in_channels = out_channels

        self.convs = nn.ModuleList(convs)
        self.conv1 = weight_norm(nn.Conv2d(kernel_size=(5, 1),
                                           in_channels=in_channels,
                                           out_channels=1024,
                                           padding=2))
        self.relu = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv2d(kernel_size=(3, 1),
                                           in_channels=1024,
                                           out_channels=1,
                                           padding=1))

    def forward(self, x):
        feature_maps = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            feature_maps.append(x)
        x = self.conv1(x)
        feature_maps.append(x)
        x = self.relu(x)
        x = self.conv2(x)
        feature_maps.append(x)
        return x, feature_maps
