import torch
from torch import Tensor


class GeneratorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_outputs, **batch):
        l = [torch.mean((discriminator_output - 1) ** 2) for discriminator_output in discriminator_outputs]
        loss_sum = torch.sum(torch.stack(l))
        # return l, loss_sum
        return loss_sum
