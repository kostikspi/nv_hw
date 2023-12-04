import torch
from torch import Tensor


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_outputs, discriminator_real_outputs, **batch):
        l_r = [torch.mean((discriminator_output - 1) ** 2) for discriminator_output in discriminator_real_outputs]
        l_g = [torch.mean(discriminator_output ** 2) for discriminator_output in discriminator_outputs]
        loss_sum = torch.sum(torch.stack(l_r)) + torch.sum(torch.stack(l_g))
        # return l_r, l_g, loss_sum
        return loss_sum
