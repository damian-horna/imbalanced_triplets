from collections import Counter
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from utils import TripletLossWeighted, TripletLoss, TripletCleveland, calc_embeddings


def config_calculation_strategy1(datasets):
    config = {}
    for dataset_name in datasets:
        data, target = datasets[dataset_name]['data'], datasets[dataset_name]['target']
        neural_net_config = {
            "units_1st_layer": data.shape[1],
            "units_2nd_layer": data.shape[1] * 2,
            "units_3rd_layer": data.shape[1],
            "units_latent_layer": data.shape[1] // 2
        }
        config[dataset_name] = {
            "nn_config": neural_net_config,
            "weighted_triplet_loss": True
        }
    return config


def config_calculation_strategy2(datasets):
    config = {}
    for dataset_name in datasets:
        data, target = datasets[dataset_name]['data'], datasets[dataset_name]['target']
        neural_net_config = {
            "units_1st_layer": data.shape[1],
            "units_2nd_layer": max(16, data.shape[1] * 3),
            "units_3rd_layer": max(data.shape[1] * 2, 8),
            "units_latent_layer": max(4, data.shape[1] // 2)
        }
        config[dataset_name] = {
            "nn_config": neural_net_config,
            "weighted_triplet_loss": True,
            "lr": 1e-4,
            "batch_size": 32,
            "gamma": 0.95,
            "epochs": 100
        }
    return config


def weights_calculation_strategy1(X_train, y_train):
    cards = Counter(y_train)
    # weights = {c: (1/v) * 100 for c,v in cards.items()}
    weights = {c: 1/v for c, v in cards.items()}
    weights_normalized = {c: weights[c]/sum(weights.values()) for c in cards.keys()}
    print(f"Class cardinalities: {cards}")
    print(f"Weights: {weights_normalized}")
    return weights_normalized


class EmbeddingNet(nn.Module):
    def __init__(self, nn_config):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(nn_config["units_1st_layer"], nn_config["units_2nd_layer"]),
                                nn.ReLU(),
                                nn.Linear(nn_config["units_2nd_layer"], nn_config["units_3rd_layer"]),
                                nn.ReLU(),
                                nn.Linear(nn_config["units_3rd_layer"], nn_config["units_latent_layer"])
                                )

    def forward(self, x):
        output = self.fc(x)
        return output

    def embed(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def embed(self, x):
        return self.embedding_net(x)


def train_tripletnet(model, device, train_loader, optimizer, epoch, weights, nn_config, log_interval=10, dry_run=False):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data[0] = torch.reshape(data[0], (data[0].shape[0], nn_config["units_1st_layer"]))
        data[1] = torch.reshape(data[1], (data[0].shape[0], nn_config["units_1st_layer"]))
        data[2] = torch.reshape(data[2], (data[0].shape[0], nn_config["units_1st_layer"]))
        data = tuple(d.cuda() for d in data)

        optimizer.zero_grad()
        outputs = model(*data)
        loss_inputs = outputs
        if weights:
            target_weights = []
            for t in target[0]:
                target_weights.append(weights[t.item()])
            w = torch.Tensor(np.array(target_weights)).to(device)
            loss_fn = TripletLossWeighted(1.0, weights=w)
        else:
            loss_fn = TripletLoss(1.0)

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))
        train_loss.append(loss.item())
    return np.mean(train_loss)


def test_tripletnet(model, device, test_loader, weights, nn_config):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data[0] = torch.reshape(data[0], (data[0].shape[0], nn_config["units_1st_layer"]))
            data[1] = torch.reshape(data[1], (data[0].shape[0], nn_config["units_1st_layer"]))
            data[2] = torch.reshape(data[2], (data[0].shape[0], nn_config["units_1st_layer"]))
            data = tuple(d.cuda() for d in data)

            outputs = model(*data)

            loss_inputs = outputs
            if weights:
                target_weights = []
                for t in target[0]:
                    target_weights.append(weights[t.item()])
                w = torch.Tensor(np.array(target_weights)).to(device)
                loss_fn = TripletLossWeighted(1.0, weights=w)
            else:
                loss_fn = TripletLoss(1.0)

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            test_loss.append(loss.item())
    # print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return np.mean(test_loss)


def train_triplets(X_train, y_train, X_test, y_test, weights, cfg):
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed = 7
    batch_size = cfg["batch_size"]
    test_batch_size = cfg["batch_size"]
    use_cuda = True
    lr = cfg["lr"]
    gamma = cfg["gamma"]
    epochs = cfg["epochs"]
    save_model = True
    log_interval = 20
    nn_config = cfg["nn_config"]

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    dataset1.train_data = torch.Tensor(X_train)
    dataset1.train_labels = torch.Tensor(y_train)
    dataset1.train = True

    dataset2 = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    dataset2.test_data = torch.Tensor(X_test)
    dataset2.test_labels = torch.Tensor(y_test)
    dataset2.train = False

    triplet_train_dataset = TripletCleveland(dataset1)
    triplet_test_dataset = TripletCleveland(dataset2)

    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, **train_kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, **test_kwargs)

    embedding_net = EmbeddingNet(nn_config)
    model = TripletNet(embedding_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    test_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        train_losses.append(train_tripletnet(model, device, triplet_train_loader, optimizer, epoch, weights, nn_config, log_interval))
        test_losses.append(test_tripletnet(model, device, triplet_test_loader, weights, nn_config))
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn_triplet.pt")

    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)

    embeddings_train, _ = calc_embeddings(model, device, train_loader)
    embeddings_test, _ = calc_embeddings(model, device, test_loader)
    plt.plot(test_losses, label="test losses")
    plt.plot(train_losses, label="train losses")
    plt.legend()
    plt.show()

    return embeddings_train, embeddings_test