import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

# https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch


class Simulation_old(Dataset):
    def __init__(self, df, num_features, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToTensor()
        self.num_features = num_features
        if self.is_train:
            self.images = df.iloc[:, 3:].values  # .astype(np.uint8)
            self.label_s = df.iloc[:, 1].values
            self.label_u = df.iloc[:, 2].values
            self.index = df.index.values
        else:
            # self.images = df.values #.astype(np.uint8)
            self.images = df.iloc[:, 3:].values  # .astype(np.uint8)
            self.label_s = df.iloc[:, 1].values
            self.label_u = df.iloc[:, 2].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        global anchor_label
        anchor_img = self.images[item].reshape(1, self.num_features, 1)

        if self.is_train:
            anchor_label = self.label_s[item]
            anchor_label_u = self.label_u[item]

            positive_list = self.index[self.index != item][self.label_s[self.index != item] == anchor_label]
            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(1, self.num_features, 1)

            negative_list = self.index[self.index != item][self.label_s[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(1, self.num_features, 1)

            if self.transform:
                anchor_img = self.transform(anchor_img.astype(np.double))
                positive_img = self.transform(positive_img.astype(np.double))
                negative_img = self.transform(negative_img.astype(np.double))
            return anchor_img, positive_img, negative_img, anchor_label, anchor_label_u

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img.astype(np.double))
                anchor_label = self.label_s[item]
                anchor_label_u = self.label_u[item]
            return anchor_img, anchor_label, anchor_label_u


class Simulation(Dataset):
    def __init__(self, df, num_features, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToTensor()
        self.num_features = num_features
        if self.is_train:
            self.images = df.iloc[:, 3:].values  # .astype(np.uint8)
            self.label_s = df.iloc[:, 1].values
            self.label_u = df.iloc[:, 2].values
            self.index = df.index.values
        else:
            # self.images = df.values #.astype(np.uint8)
            self.images = df.iloc[:, 3:].values  # .astype(np.uint8)
            self.label_s = df.iloc[:, 1].values
            self.label_u = df.iloc[:, 2].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        global anchor_label
        anchor_img = self.images[item].reshape(1, self.num_features, 1)
        anchor_label = self.label_s[item]
        anchor_label_u = self.label_u[item]

        positive_list = self.index[self.index != item][self.label_s[self.index != item] == anchor_label]
        positive_item = random.choice(positive_list)
        positive_img = self.images[positive_item].reshape(1, self.num_features, 1)

        negative_list = self.index[self.index != item][self.label_s[self.index != item] != anchor_label]
        negative_item = random.choice(negative_list)
        negative_img = self.images[negative_item].reshape(1, self.num_features, 1)

        if self.transform:
            anchor_img = self.transform(anchor_img.astype(np.double))
            positive_img = self.transform(positive_img.astype(np.double))
            negative_img = self.transform(negative_img.astype(np.double))
        return anchor_img, positive_img, negative_img, anchor_label, anchor_label_u

