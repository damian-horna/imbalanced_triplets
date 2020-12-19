from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(7)

            triplets = []

            for i in range(len(self.test_data)):
                positive = random_state.choice(self.label_to_indices[self.test_labels[i].item()])
                rand_other = np.random.choice(list(self.labels_set - {self.test_labels[i].item()}))
                negative = random_state.choice(self.label_to_indices[rand_other])
                t = [i, positive, negative]
                triplets.append(t)

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            anchor, anchor_label = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[anchor_label])
            negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            positive = self.train_data[positive_index]
            negative = self.train_data[negative_index]
        else:
            anchor = self.test_data[self.test_triplets[index][0]]
            positive = self.test_data[self.test_triplets[index][1]]
            negative = self.test_data[self.test_triplets[index][2]]

        anchor = Image.fromarray(anchor.numpy(), mode='L')
        positive = Image.fromarray(positive.numpy(), mode='L')
        negative = Image.fromarray(negative.numpy(), mode='L')
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return (anchor, positive, negative), []

    def __len__(self):
        return len(self.mnist_dataset)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def plot_embeddings(X_embedded, labels):
    embedded_df = pd.DataFrame.from_dict({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "color": labels
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=embedded_df, x="x", y="y", hue="color", palette="deep")


def calc_embeddings(model, device, loader):
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)
            embedding = model.embed(data)
            embeddings.append(embedding[0].cpu().numpy())
            labels.append(target[0].cpu().numpy())

    return np.array(embeddings), np.array(labels)


class TripletCleveland(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.train = self.ds.train

        if self.train:
            self.train_labels = self.ds.train_labels
            self.train_data = self.ds.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.ds.test_labels
            self.test_data = self.ds.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(7)

            triplets = []

            for i in range(len(self.test_data)):
                positive = random_state.choice(self.label_to_indices[self.test_labels[i].item()])
                rand_other = np.random.choice(list(self.labels_set - {self.test_labels[i].item()}))
                negative = random_state.choice(self.label_to_indices[rand_other])
                t = [i, positive, negative]
                triplets.append(t)

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            anchor, anchor_label = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[anchor_label])
            negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            positive = self.train_data[positive_index]
            negative = self.train_data[negative_index]
        else:
            anchor = self.test_data[self.test_triplets[index][0]]
            positive = self.test_data[self.test_triplets[index][1]]
            negative = self.test_data[self.test_triplets[index][2]]

        return (anchor, positive, negative), []

    def __len__(self):
        return len(self.ds)