import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        loss = self.base_loss(pred, target)

        target = (target > 0.01).float()

        pos_loss = (loss * target).sum() / (target.sum() + 1e-6)
        neg_loss = (loss * (1 - target)).sum() / ((1 - target).sum() + 1e-6)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss
