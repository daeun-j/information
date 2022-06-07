import torch.nn as nn
import torch
from torch.nn import functional as F
from sklearn.metrics import recall_score, precision_score, accuracy_score


class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        # return (x1 - x2).pow(2).sum()
        return (x1 - x2).pow(2).sum()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(abs(distance_positive - distance_negative + self.margin))
        return losses.mean()


class TripletLoss_hz(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss_hz, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        # return (x1 - x2).pow(2).sum()
        return (x1 - x2).pow(2).sum()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, anchor_z: torch.Tensor, positive_z: torch.Tensor, negative_z: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        distance_positive_z = self.calc_euclidean(anchor_z, positive_z)
        distance_negative_z = self.calc_euclidean(anchor_z, negative_z)
        losses = torch.relu(abs(distance_positive - distance_negative + self.margin))
        losses_z = torch.relu(abs(distance_positive_z - distance_negative_z + self.margin))

        return losses.mean() + losses_z.mean()


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, num_features):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, num_features), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = - 0.5 * torch.sum(1 + logvar*2 - mu.pow(2) - logvar.exp().pow(2))
    KLD = 0.5 * torch.sum(mu+logvar)
    #-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE+KLD


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


def evaluate_class(y, y_hat):
    # confusion_matrix(y, y_hat, labels=[0, 1])
    # confusion_matrix(y, y_hat, labels=[0, 1, 2])

    #print(classification_report(y, y_hat))
    # print("y, y_hat", y, y_hat)
    #print("Accracy {}  | macro precision {}| macro recall {}".format(accuracy_score(y, y_hat),precision_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='macro'))
    return accuracy_score(y, y_hat), precision_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='macro')
