import torch
import torch.nn as nn

from torch.nn.utils import weight_norm


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv = weight_norm(nn.Conv1d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel,
                                          dilation=dilation,
                                          padding=(kernel * dilation - dilation) // 2))

    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation):
        super().__init__()
        self.cnn_block1 = nn.Sequential(*nn.ModuleList([ConvReLU(in_channels=in_channels,
                                                                 out_channels=out_channels,
                                                                 kernel=kernel,
                                                                 dilation=d_i) for d_i in dilation[0]]))
        self.cnn_block2 = nn.Sequential(*nn.ModuleList([ConvReLU(in_channels=in_channels,
                                                                 out_channels=out_channels,
                                                                 kernel=kernel,
                                                                 dilation=d_i) for d_i in dilation[1]]))
        self.cnn_block3 = nn.Sequential(*nn.ModuleList([ConvReLU(in_channels=in_channels,
                                                                 out_channels=out_channels,
                                                                 kernel=kernel,
                                                                 dilation=d_i) for d_i in dilation[2]]))

    def forward(self, x):
        res = x
        x = self.cnn_block1(x)
        x = x + res
        res = x
        x = self.cnn_block2(x)
        x = x + res
        res = x
        x = self.cnn_block3(x)
        x = x + res

        return x
