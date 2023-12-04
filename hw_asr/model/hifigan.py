from typing import Union

import torch.nn.functional
from torch import nn, Tensor
from torch.nn import Sequential

from hw_asr.base import BaseModel

from hw_asr.model.generator import Generator
from hw_asr.model.MPD import MPD
from hw_asr.model.MSD import MSD


class HiFiGAN(BaseModel):
    def __init__(self, kernels_u=None, dilations=None, out_channels=None, kernels_r=None):
        super().__init__()
        self.generator = Generator(kernels_u=kernels_u,
                                   dilations=dilations,
                                   out_channels=out_channels,
                                   kernels_r=kernels_r)
        self.mpd = nn.ModuleList([MPD(2), MPD(3), MPD(5), MPD(7), MPD(11)])
        self.msd = nn.ModuleList([MSD(0), MSD(1), MSD(2)])

    def forward(self, spectrogram, wave, **batch) -> Union[Tensor, dict]:
        x = self.generator(spectrogram)
        mpd_feature_maps = []
        mpd_outputs = []
        for discriminator in self.mpd:
            disc_input = self.pad_reshape(x, discriminator.period)
            output, feature_map = discriminator(disc_input)
            mpd_feature_maps.append(feature_map)
            mpd_outputs.append(output)
        msd_feature_maps = []
        msd_outputs = []
        for discriminator in self.msd:
            disc_input = self.pool(x, discriminator.pool)
            output, feature_map = discriminator(disc_input)
            msd_feature_maps.append(feature_map)
            msd_outputs.append(output)
        mpd_real_feature_maps = []
        mpd_real_outputs = []
        for discriminator in self.mpd:
            disc_input = self.pad_reshape(wave, discriminator.period)
            output, feature_map = discriminator(disc_input)
            mpd_real_feature_maps.append(feature_map)
            mpd_real_outputs.append(output)
        msd_real_feature_maps = []
        msd_real_outputs = []
        for discriminator in self.msd:
            disc_input = self.pool(wave, discriminator.pool)
            output, feature_map = discriminator(disc_input)
            msd_real_feature_maps.append(feature_map)
            msd_real_outputs.append(output)
        return {"gen_audio": x,
                "mpd_feature_maps": mpd_feature_maps,
                "mpd_outputs": mpd_outputs,
                "msd_feature_maps": msd_feature_maps,
                "msd_outputs": msd_outputs,
                "mpd_real_feature_maps": mpd_real_feature_maps,
                "mpd_real_outputs": mpd_real_outputs,
                "msd_real_feature_maps": msd_real_feature_maps,
                "msd_real_outputs": msd_real_outputs}
    def transform_input_lengths(self, input_lengths):
        return input_lengths

    @staticmethod
    def pad_reshape(x, period):
        x = torch.nn.functional.pad(x, (0, (period - (x.shape[2] % period)) % period), "reflect")
        x = x.view(x.shape[0], x.shape[1], (x.shape[2] + ((period - (x.shape[2] % period)) % period)) // period, period)
        return x

    @staticmethod
    def pool(x, pool):
        for i in range(pool):
            x = torch.nn.functional.avg_pool1d(x, 4, 2, padding=2)
        return x


