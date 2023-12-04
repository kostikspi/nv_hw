import torch


class MelSpectrogramLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, target_mel, predict_mel, **batch):
        loss = self.l1_loss(target_mel, predict_mel)
        return loss
