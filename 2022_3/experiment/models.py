import torch.nn as nn
import torch
import torch.utils.data
from torch.nn import functional as F
from args import *

####################
######## H #########
####################
args = parser.parse_args()


class Network(nn.Module):
    def __init__(self, num_features, emb_dim=2):
        super(Network, self).__init__()
        self.num_features = num_features
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 1, 1),
            nn.PReLU(),
            # nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.PReLU(),
            nn.Linear(8, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x.reshape(1, 8, 1))
        # print(x)
        x = x.view(-1, 1)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class Network2(nn.Module):
    def __init__(self, num_features, emb_dim=2):
        super(Network2, self).__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=2, dilation=2,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=4, dilation=2,
                               kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=6, dilation=2,
                               kernel_size=7, stride=1)
        self.activation = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(8, 4),
            nn.PReLU(),
            nn.Linear(4, emb_dim)
        )

    def forward(self, x):
        x_raw = x.reshape(1, self.num_features, 1)
        x = self.conv1(x.reshape(1, self.num_features, 1))
        x = self.conv2(x.reshape(1, self.num_features, 1))
        x = self.conv3(x.reshape(1, self.num_features, 1))
        x = self.activation(torch.add(x, x_raw))
        x = x.view(1, -1)
        x = self.fc(x)

        return x


class Network3(nn.Module):
    def __init__(self, num_features):
        super(Network3, self).__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=2, dilation=2,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=4, dilation=2,
                               kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=6, dilation=2,
                               kernel_size=7, stride=1)
        self.activation = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(8, num_features)
        )

    def forward(self, x):
        x_raw = x.reshape(1, self.num_features, 1)
        x = self.conv1(x.reshape(1, self.num_features, 1))
        x = self.conv2(x.reshape(1, self.num_features, 1))
        x = self.conv3(x.reshape(1, self.num_features, 1))
        x = self.activation(torch.add(x, x_raw))
        x = x.view(1, -1)
        x = self.fc(x)
        return x


class Clfnet(nn.Module):
    def __init__(self, num_features):
        super(Clfnet, self).__init__()
        self.num_features = num_features
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.PReLU()
        )

    def forward(self, x):
        # x = x.reshape(1, self.num_features, 1)
        print(x, x.size())
        # x = self.fc(x)
        x = F.relu(self.fc(x))
        print(x, x.size())
        return x


class Network_bn(nn.Module):
    def __init__(self, num_features, emb_dim, batch):
        super(Network_bn, self).__init__()
        self.num_features = num_features
        self.num_batch = batch
        self.emb_dim = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=2, dilation=2,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=4, dilation=2,
                               kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=6, dilation=2,
                               kernel_size=7, stride=1)
        self.activation = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(8, 6),
            nn.PReLU(),
            nn.Linear(6, self.emb_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(self.num_features)
        self.bn2 = torch.nn.BatchNorm1d(self.num_features)

    def forward(self, x):
        x_raw = x.reshape(1, self.num_features, self.num_batch)
        x = self.conv1(x.reshape(1, self.num_features, self.num_batch))
        x = self.bn1(x)
        x = self.conv2(x.reshape(1, self.num_features, self.num_batch))
        x = self.bn2(x)
        x = self.conv3(x.reshape(1, self.num_features, self.num_batch))
        x = self.activation(torch.add(x, x_raw))
        x = x.view(self.num_batch, -1)
        x = self.fc(x)
        return x


####################
######## Z #########
####################

class VAE(nn.Module):
    def __init__(self, emb_dim):
        super(VAE, self).__init__()
        self.half_emb_dim = int(emb_dim/2)
        self.fc1 = nn.Linear(8, 4)
        self.fc21 = nn.Linear(4, self.half_emb_dim)
        self.fc22 = nn.Linear(4, self.half_emb_dim)
        self.fc3 = nn.Linear(self.half_emb_dim, 4)
        self.fc4 = nn.Linear(4, 8)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))  # F.relu
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 8))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




####################
####### h+z ########
####################

class concat_v2(nn.Module):
    def __init__(self, num_features, emb_dim, batch):
        super(concat_v2, self).__init__()
        self.num_features = num_features
        self.num_batch = batch
        self.emb_dim = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=2, dilation=2,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=4, dilation=2,
                               kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, padding=6, dilation=2,
                               kernel_size=7, stride=1)
        self.th = nn.Threshold(0.0001, 0) # less than 0.5 is 0
        self.activation = nn.PReLU()
        self.fc = nn.Sequential(
            nn.Linear(self.num_features,  self.num_features*2),
            nn.PReLU()
        )
        self.bn1 = torch.nn.BatchNorm1d(self.num_features)
        self.bn2 = torch.nn.BatchNorm1d(self.num_features)

    def forward(self, x):
        h1 = self.h(x)
        z1 = self.z(x, h1)
        h1 = self.activation(h1)
        # print("h, z", h1, z1)
        z1 = self.activation(z1)
        return h1, z1

    def h(self, x):
        x = x.view(self.num_batch, -1)
        # x = x.reshape(1, self.num_features, self.num_batch)
        x = self.conv1(x.reshape(1, self.num_features, self.num_batch))
        # x = self.bn1(x)
        x = self.conv2(x.reshape(1, self.num_features, self.num_batch))
        # x = self.bn2(x)
        x = self.conv3(x.reshape(1, self.num_features, self.num_batch))
        x = x.view(self.num_batch, -1)
        return x

    def z(self, x, h):
        x = x.view(self.num_batch, -1)
        # h = h.view(self.num_batch, -1)
        # x = x.reshape(1, self.num_features, self.num_batch)
        # h = h.reshape(1, self.num_features, self.num_batch)

        x = torch.sub(x, h, alpha=1)
        x = self.th(x)
        # x = x.view(self.num_batch, -1)
        x = self.fc(x)
        x = x.view(self.num_batch, -1)
        return x
