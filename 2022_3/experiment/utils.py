import torch.nn as nn
import torch


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        # return (x1 - x2).pow(2).sum()
        return (x1 - x2).pow(2).sum()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class Clfnetloss(nn.Module):
    def __init__(self):
        super(Clfnetloss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, anchor_label: torch.Tensor) -> torch.Tensor:
        pred_anchor = anchor
        pred_pos = positive
        # print(self.ce_loss(pred_anchor, anchor_label))
        losses = self.kl_loss(pred_anchor, anchor_label) + self.kl_loss(pred_pos, anchor_label)
        return losses.mean()
