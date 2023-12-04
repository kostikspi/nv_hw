import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class MSD(nn.Module):
    def __init__(self, pool, in_channels=None, out_channels=None):
        super().__init__()
        self.pool = pool
        self.conv1 = weight_norm(nn.Conv1d(kernel_size=15, in_channels=1, out_channels=16))
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = weight_norm(nn.Conv1d(in_channels=16,
                                           out_channels=64,
                                           kernel_size=41,
                                           stride=4,
                                           groups=4))
        self.conv3 = weight_norm(nn.Conv1d(in_channels=64,
                                           out_channels=256,
                                           kernel_size=41,
                                           stride=4,
                                           groups=16))
        self.conv4 = weight_norm(nn.Conv1d(in_channels=256,
                                           out_channels=1024,
                                           kernel_size=41,
                                           stride=4,
                                           groups=64))
        self.conv5 = weight_norm(nn.Conv1d(in_channels=1024,
                                           out_channels=1024,
                                           kernel_size=41,
                                           stride=4,
                                           groups=256))
        self.conv6 = weight_norm(nn.Conv1d(in_channels=1024,
                                           out_channels=1024,
                                           kernel_size=5))
        self.conv7 = weight_norm(nn.Conv1d(in_channels=1024,
                                           out_channels=1,
                                           kernel_size=3))

    def forward(self, x):
        feature_map = []
        x = self.conv1(x)
        x = self.leaky_relu(x)
        feature_map.append(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        feature_map.append(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        feature_map.append(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        feature_map.append(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        feature_map.append(x)
        x = self.conv6(x)
        x = self.leaky_relu(x)
        feature_map.append(x)
        x = self.conv7(x)
        feature_map.append(x)
        return x, feature_map
