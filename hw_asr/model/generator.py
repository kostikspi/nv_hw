import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from hw_asr.model.ResBlock import ResBlock


class Generator(nn.Module):
    def __init__(self, kernels_u, dilations, out_channels=None, kernels_r=None):
        super().__init__()
        convs = []
        mrfs = []
        in_channels = out_channels
        self.conv1 = weight_norm(nn.Conv1d(80, out_channels, 7, 1, padding=3))
        for i in range(len(kernels_u)):
            out_channels_curr = out_channels // 2 ** (i + 1)
            conv = weight_norm(nn.ConvTranspose1d(in_channels=in_channels,
                                                  out_channels=out_channels_curr,
                                                  kernel_size=kernels_u[i],
                                                  stride=kernels_u[i] // 2,
                                                  padding=(kernels_u[i] - kernels_u[i] // 2) // 2))
            mrf = nn.ModuleList([ResBlock(in_channels=out_channels_curr,
                                          out_channels=out_channels_curr,
                                          kernel=kernels_r[k],
                                          dilation=dilations[k]) for k in range(len(kernels_r))])
            in_channels = out_channels_curr

            convs.append(conv)
            mrfs.append(mrf)
        self.convs = nn.ModuleList(convs)
        self.mrfs = nn.ModuleList(mrfs)
        self.conv_out = weight_norm(nn.Conv1d(in_channels=in_channels,
                                              out_channels=1,
                                              kernel_size=7,
                                              stride=1,
                                              padding=3))
        self.tanh = nn.Tanh()
        # self.conv1 = nn.ConvTranspose2d(in_channels=1,
        #                                 out_channels=out_channels // 2,
        #                                 kernel_size=(kernels_u[0], 1),
        #                                 dilation=1, stride=kernels_u[0] // 2)
        # self.generator_block1 = nn.ModuleList([ResBlock(in_channels=None,
        #                                                 out_channels=None,
        #                                                 kernel=kernels_r[k],
        #                                                 dilation=dilations[k]) for k in range(len(kernels_r))])
        #
        # self.conv2 = nn.ConvTranspose2d(in_channels=1,
        #                                 out_channels=out_channels[0],
        #                                 kernel_size=kernels_u[0],
        #                                 dilation=1, stride=kernels_u[0] // 2)
        #
        # self.generator_block2 = nn.ModuleList([ResBlock(in_channels=None,
        #                                                 out_channels=None,
        #                                                 kernel=kernels_r[k],
        #                                                 dilation=dilations[k]) for k in range(len(kernels_r))])
        #
        # self.conv3 = nn.ConvTranspose2d(in_channels=1,
        #                                 out_channels=out_channels[0],
        #                                 kernel_size=kernels_u[0],
        #                                 dilation=1, stride=kernels_u[0] // 2)
        #
        # self.generator_block3 = nn.ModuleList([ResBlock(in_channels=None,
        #                                                 out_channels=None,
        #                                                 kernel=kernels_r[k],
        #                                                 dilation=dilations[k]) for k in range(len(kernels_r))])
        #
        # self.conv4 = nn.ConvTranspose2d(in_channels=1,
        #                                 out_channels=out_channels[0],
        #                                 kernel_size=kernels_u[0],
        #                                 dilation=1, stride=kernels_u[0] // 2)
        #
        # self.generator_block4 = nn.ModuleList([ResBlock(in_channels=None,
        #                                                 out_channels=None,
        #                                                 kernel=kernels_r[k],
        #                                                 dilation=dilations[k]) for k in range(len(kernels_r))])

    def forward(self, x):
        # batch  x n_feats x time
        x = self.conv1(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            out = None
            for j, module in enumerate(self.mrfs[i]):
                if j == 0:
                    out = module(x)
                else:
                    out = out + module(x)
            x = out / len(self.convs)

        x = self.tanh(self.conv_out(x))

        return x
