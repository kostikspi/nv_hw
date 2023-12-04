import torch
from torch import Tensor


class FeatureMatchingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_feature_maps, generator_feature_maps, **batch):
        feature_matching_loss = []
        for i in range(len(real_feature_maps)):
            for j in range(len(real_feature_maps[i])):
                feature_matching_loss.append(torch.mean(torch.abs(generator_feature_maps[i][j] - real_feature_maps[i][j])))
        return torch.sum(torch.stack(feature_matching_loss))

