from collections import Counter
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import random
from utils import calc_embeddings, plot_embeddings
import pandas as pd
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from itertools import combinations
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

dt_name_to_cols_to_encode = {
    'cmc': [1,2,6,7],
    'dermatology': list(range(0,10)) + list(range(10,33)),
    'hayes-roth':[0,1,2,3],
    'new_vehicle': [],
    'new_yeast': [5],
    '1czysty-cut': [],
    '2delikatne-cut': [],
    '3mocniej-cut': [],
    '4delikatne-bezover-cut': [],
    'balance-scale': [0,1,2,3],
    'cleveland': [2,6,10,11,12],
    'cleveland_v2': [2, 6, 10, 11, 12],
    'glass': [],
    'new_ecoli': [],
    'new_led7digit': [],
    'new_winequality-red': [],
    'thyroid-newthyroid': []
}

dt_name_minority_classes = {
    '1czysty-cut': [1, 2],
    '2delikatne-cut': [1,2],
    '3mocniej-cut': [1,2],
    '4delikatne-bezover-cut': [1,2],
    'balance-scale': [0],
    'cleveland': [1,2,3,4],
    'cleveland_v2': [1,2,3],
    'cmc': [1],
    'dermatology': [5],
    'glass': [5,2,4],
    'hayes-roth': [2],
    'new_vehicle': [0,2],
    'new_yeast': [2,3,4,5,6],
    'new_ecoli': [4,2,3],
    'new_led7digit': [1,4],
    'new_winequality-red': [3,2],
    'thyroid-newthyroid': [2,1]
}


def config_tuned_for_lda(config):
    config['cmc'] = {'nn_config': {'units_1st_layer': 17,
                                   'units_2nd_layer': 256,
                                   'units_3rd_layer': 128,
                                   'units_latent_layer': 8},
                     'weighted_triplet_loss': True,
                     'lr': 0.0001,
                     'batch_size': 16,
                     'gamma': 0.99,
                     'epochs': 100}

    config['dermatology'] = {'nn_config': {'units_1st_layer': 97,
                                           'units_2nd_layer': 512,
                                           'units_3rd_layer': 256,
                                           'units_latent_layer': 16},
                             'weighted_triplet_loss': True,
                             'lr': 0.0015,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 150}

    config['hayes-roth'] = {'nn_config': {'units_1st_layer': 11,
                                          'units_2nd_layer': 128,
                                          'units_3rd_layer': 64,
                                          'units_latent_layer': 16},
                            'weighted_triplet_loss': True,
                            'lr': 0.0015,
                            'batch_size': 16,
                            'gamma': 0.99,
                            'epochs': 300}

    config['new_vehicle'] = {'nn_config': {'units_1st_layer': 18,
                                           'units_2nd_layer': 256,
                                           'units_3rd_layer': 128,
                                           'units_latent_layer': 16},
                             'weighted_triplet_loss': True,
                             'lr': 0.003,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 100}

    config['new_yeast'] = {'nn_config': {'units_1st_layer': 9,
                                         'units_2nd_layer': 300,
                                         'units_3rd_layer': 200,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0004,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 100}

    config['balance-scale'] = {'nn_config': {'units_1st_layer': 16,
                                             'units_2nd_layer': 256,
                                             'units_3rd_layer': 128,
                                             'units_latent_layer': 10},
                               'weighted_triplet_loss': True,
                               'lr': 0.004,
                               'batch_size': 16,
                               'gamma': 0.99,
                               'epochs': 200}

    config['cleveland'] = {'nn_config': {'units_1st_layer': 24,
                                         'units_2nd_layer': 72,
                                         'units_3rd_layer': 48,
                                         'units_latent_layer': 16},
                           'weighted_triplet_loss': True,
                           'lr': 0.0005,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 150}

    config['cleveland_v2'] = {'nn_config': {'units_1st_layer': 23,
                                            'units_2nd_layer': 256,
                                            'units_3rd_layer': 128,
                                            'units_latent_layer': 16},
                              'weighted_triplet_loss': True,
                              'lr': 0.0005,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 100}

    config['glass'] = {'nn_config': {'units_1st_layer': 9,
                                     'units_2nd_layer': 256,
                                     'units_3rd_layer': 128,
                                     'units_latent_layer': 12},
                       'weighted_triplet_loss': True,
                       'lr': 0.0005,
                       'batch_size': 16,
                       'gamma': 0.99,
                       'epochs': 150}

    config['thyroid-newthyroid'] = {'nn_config': {'units_1st_layer': 5,
                                                  'units_2nd_layer': 64,
                                                  'units_3rd_layer': 32,
                                                  'units_latent_layer': 8},
                                    'weighted_triplet_loss': True,
                                    'lr': 0.004,
                                    'batch_size': 16,
                                    'gamma': 0.99,
                                    'epochs': 150}

    config['new_ecoli'] = {'nn_config': {'units_1st_layer': 7,
                                         'units_2nd_layer': 128,
                                         'units_3rd_layer': 64,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0005,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 100}

    config['new_led7digit'] = {'nn_config': {'units_1st_layer': 7,
                                             'units_2nd_layer': 64,
                                             'units_3rd_layer': 32,
                                             'units_latent_layer': 8},
                               'weighted_triplet_loss': True,
                               'lr': 0.001,
                               'batch_size': 16,
                               'gamma': 0.99,
                               'epochs': 100}

    config['new_winequality-red'] = {'nn_config': {'units_1st_layer': 11,
                                                   'units_2nd_layer': 128,
                                                   'units_3rd_layer': 64,
                                                   'units_latent_layer': 12},
                                     'weighted_triplet_loss': True,
                                     'lr': 0.003,
                                     'batch_size': 16,
                                     'gamma': 0.99,
                                     'epochs': 100}

    config['4delikatne-bezover-cut'] = {'nn_config': {'units_1st_layer': 2,
                                                      'units_2nd_layer': 128,
                                                      'units_3rd_layer': 64,
                                                      'units_latent_layer': 8},
                                        'weighted_triplet_loss': True,
                                        'lr': 0.003,
                                        'batch_size': 16,
                                        'gamma': 0.99,
                                        'epochs': 100}

    config['3mocniej-cut'] = {'nn_config': {'units_1st_layer': 2,
                                            'units_2nd_layer': 128,
                                            'units_3rd_layer': 64,
                                            'units_latent_layer': 10},
                              'weighted_triplet_loss': True,
                              'lr': 0.003,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 100}

    config['1czysty-cut'] = {'nn_config': {'units_1st_layer': 2,
                                           'units_2nd_layer': 64,
                                           'units_3rd_layer': 32,
                                           'units_latent_layer': 8},
                             'weighted_triplet_loss': True,
                             'lr': 0.003,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 100}

    config['2delikatne-cut'] = {'nn_config': {'units_1st_layer': 2,
                                              'units_2nd_layer': 128,
                                              'units_3rd_layer': 64,
                                              'units_latent_layer': 12},
                                'weighted_triplet_loss': True,
                                'lr': 0.003,
                                'batch_size': 16,
                                'gamma': 0.99,
                                'epochs': 100}
    return config


def one_hot_encode_all(datasets):
    ds_names = list(dt_name_to_cols_to_encode.keys())

    for ds_name in ds_names:
        k = ds_name
        df = pd.DataFrame(data=datasets[k]['data'])
        encoded = pd.get_dummies(df, columns=dt_name_to_cols_to_encode[ds_name], drop_first=True)

        datasets[f"{k}_encoded"] = {'data': encoded.values, 'target': datasets[k]['target']}
    return datasets


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


def config_calculation_strategy3(datasets):
    config = {}
    for dataset_name in datasets:
        data, target = datasets[dataset_name]['data'], datasets[dataset_name]['target']
        neural_net_config = {
            "units_1st_layer": data.shape[1],
            "units_2nd_layer": max(16, data.shape[1] * 3),
            "units_3rd_layer": max(data.shape[1] * 2, 8),
            "units_latent_layer": max(4, data.shape[1] // 2),
            "units_decision_layer": np.unique(target).size
        }
        config[dataset_name] = {
            "nn_config": neural_net_config,
            "weighted_triplet_loss": True,
            "lr": 0.003,
            "batch_size": 16,
            "gamma": 0.99,
            "epochs": 100
        }
    return config


def weights_calculation_strategy1(X_train, y_train):
    # Inverse class frequencies, normalized
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
                                nn.PReLU(),
                                nn.Linear(nn_config["units_2nd_layer"], nn_config["units_3rd_layer"]),
                                nn.PReLU(),
                                nn.Linear(nn_config["units_3rd_layer"], nn_config["units_latent_layer"])
                                )

    def forward(self, x):
        output = self.fc(x)
        return output

    def embed(self, x):
        return self.forward(x)


def train_safenessnet(model, device, train_loader, optimizer, epoch, weights, nn_config, log_interval=10, pca=None, X_train=None):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # plot_batch(X_train, batch_idx, data, pca)
        data = tuple(d.cuda() for d in data)

        target = tuple(t.cuda() for t in target)

        optimizer.zero_grad()
        outputs = model(*data)
        outputs = (outputs,)

        loss_inputs = outputs
        loss_inputs += (target,)

        loss_fn = SafenessLoss()

        loss = loss_fn(*loss_inputs)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


class SafenessLoss(nn.Module):
    def __init__(self):
        super(SafenessLoss, self).__init__()

    def forward(self, embeddings, target):
        batch_size = embeddings[0].shape[0]
        anchor = embeddings[0]
        anchor_label = target[0]

        losses = []
        for i in range(batch_size):
            emb_same_class = [emb[i,:] for emb, clazz in zip(embeddings[1:], target[1:]) if clazz[i] == anchor_label[i]]
            emb_different_class = [emb[i,:] for emb, clazz in zip(embeddings[1:], target[1:]) if clazz[i] != anchor_label[i]]

            same_class_dists = [(anchor[i, :] - emb).pow(2).sum() for emb in emb_same_class]
            different_class_dists = [(anchor[i, :] - emb).pow(2).sum() for emb in emb_different_class]

            alpha = max(same_class_dists) if same_class_dists else 1.0 # Set alpha to some predefined margin value if there are no neighbors from the same class

            same_class_dist_sum = torch.stack(same_class_dists).sum() if same_class_dists else 0

            diff_class_mins = [dist - alpha for dist in different_class_dists if dist-alpha < 0]
            different_class_dist_min_sum = torch.stack(diff_class_mins).sum() if diff_class_mins else 0

            if torch.is_tensor(same_class_dist_sum - different_class_dist_min_sum):
                losses.append(same_class_dist_sum - different_class_dist_min_sum)

        # print("Losses:")
        # print(losses)
        result = torch.stack(losses).mean()
        return result


def plot_batch(X_train, batch_idx, data, pca):
    data = np.array([t.numpy()[0] for t in data])
    print(data.shape)
    plt.figure(figsize=(8, 6))
    plt.title(f"{batch_idx}")
    plt.scatter(pca.transform(X_train)[:, 0], pca.transform(X_train)[:, 1], c='y', s=10)
    plt.scatter(pca.transform(data)[:, 0], pca.transform(data)[:, 1], marker='x', c='r', s=10)
    plt.show()


def test_safenessnet(model, device, test_loader, weights, nn_config):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for data, target in test_loader:
            data = tuple(d.cuda() for d in data)

            target = tuple(t.cuda() for t in target)

            outputs = model(*data)
            outputs = (outputs,)

            loss_inputs = outputs
            loss_inputs += (target,)

            loss_fn = SafenessLoss()

            loss = loss_fn(*loss_inputs)
            test_loss.append(loss.item())
    return np.mean(test_loss)


def train_triplets(X_train, y_train, X_test, y_test, weights, cfg, pca):
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

    dataset1, dataset2, neighbors_test_loader, neighbors_train_loader = init_loaders(X_train, X_test, y_train, y_test, X_train, X_test,
                                                                                     batch_size, test_batch_size,
                                                                                     use_cuda)

    embedding_net = EmbeddingNet(nn_config)
    model = SafenessNet(embedding_net).to(device)
    # model = embedding_net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Calculate initial embeddings:
    # embeddings_train, embeddings_test= train_and_test_embeddings(dataset1, dataset2, device, model)

    test_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        # print(f"Epoch: {epoch}")
        train_losses.append(train_safenessnet(model, device, neighbors_train_loader, optimizer, epoch, weights, nn_config, log_interval, pca, X_train))
        test_losses.append(test_safenessnet(model, device, neighbors_test_loader, weights, nn_config))
        scheduler.step()

        embeddings_train, embeddings_test = train_and_test_embeddings(dataset1, dataset2, device, model)
        dataset1, dataset2, neighbors_test_loader, neighbors_train_loader = init_loaders(X_train, X_test, y_train,
                                                                                         y_test, embeddings_train, embeddings_test,
                                                                                         batch_size, test_batch_size,
                                                                                         use_cuda)
        if epoch % 10 == 0:
            # PCA embeddings_train
            pca = PCA(n_components=2)
            plot_embeddings(pca.fit_transform(embeddings_train), y_train)
            plt.title(f"Embeddings_train after {epoch} epochs")
            plt.show()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn_triplet.pt")

    embeddings_train, embeddings_test = train_and_test_embeddings(dataset1, dataset2, device, model)
    plt.plot(test_losses, label="test losses")
    plt.plot(train_losses, label="train losses")
    plt.legend()
    plt.show()

    return embeddings_train, embeddings_test


def train_and_test_embeddings(dataset1, dataset2, device, model):
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)
    embeddings_train, _ = calc_embeddings(model, device, train_loader)
    embeddings_test, _ = calc_embeddings(model, device, test_loader)
    return embeddings_train, embeddings_test


def init_loaders(X_train, X_test, y_train, y_test, train_repr, test_repr, batch_size, test_batch_size, use_cuda):
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
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

    neighbors_train_dataset = NeighborsDataset(dataset1, train_repr)
    neighbors_test_dataset = NeighborsDataset(dataset2, test_repr)

    neighbors_train_loader = torch.utils.data.DataLoader(neighbors_train_dataset, **train_kwargs)
    neighbors_test_loader = torch.utils.data.DataLoader(neighbors_test_dataset, **test_kwargs)

    return dataset1, dataset2, neighbors_test_loader, neighbors_train_loader


class NeighborsDataset(Dataset):
    def __init__(self, ds, representation, n_neighbors=21):
        np.random.seed(0)
        self.ds = ds
        self.train = self.ds.train
        self.n_neighbors = n_neighbors
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors)
        self.representation = representation

        if self.train:
            self.train_labels = self.ds.train_labels
            self.train_data = self.ds.train_data
        else:
            self.test_labels = self.ds.test_labels
            self.test_data = self.ds.test_data
        self.neigh.fit(self.representation)

    def __getitem__(self, index):
        anchor_repr = self.representation[index, :]
        if self.train:
            anchor = self.train_data[index]
            anchor_label = self.train_labels[index]
            neigh_indices = self.neigh.kneighbors([anchor_repr], return_distance=False)
            neigh_indices = [ind for ind in neigh_indices[0] if ind != index] # without self
            neigh_indices = neigh_indices[: self.n_neighbors-1]
            neighbors = self.train_data[neigh_indices, :]
            neighbors_labels = self.train_labels[neigh_indices]
        else:
            anchor = self.test_data[index]
            anchor_label = self.test_labels[index]
            neigh_indices = self.neigh.kneighbors([anchor_repr], return_distance=False)
            neigh_indices = [ind for ind in neigh_indices[0] if ind != index] # without self
            neigh_indices = neigh_indices[: self.n_neighbors-1]
            neighbors = self.test_data[neigh_indices]
            neighbors_labels = self.test_labels[neigh_indices]
        return (anchor, *neighbors), [anchor_label, *neighbors_labels]

    def __len__(self):
        return len(self.ds)


# ---------------------------------------------- SAFENESS
class SafenessNet(nn.Module):
    def __init__(self, embedding_net):
        super(SafenessNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20):
        anchor_emb = self.embedding_net(anchor)
        n1_emb = self.embedding_net(n1)
        n2_emb = self.embedding_net(n2)
        n3_emb = self.embedding_net(n3)
        n4_emb = self.embedding_net(n4)
        n5_emb = self.embedding_net(n5)
        n6_emb = self.embedding_net(n6)
        n7_emb = self.embedding_net(n7)
        n8_emb = self.embedding_net(n8)
        n9_emb = self.embedding_net(n9)
        n10_emb = self.embedding_net(n10)
        n11_emb = self.embedding_net(n11)
        n12_emb = self.embedding_net(n12)
        n13_emb = self.embedding_net(n13)
        n14_emb = self.embedding_net(n14)
        n15_emb = self.embedding_net(n15)
        n16_emb = self.embedding_net(n16)
        n17_emb = self.embedding_net(n17)
        n18_emb = self.embedding_net(n18)
        n19_emb = self.embedding_net(n19)
        n20_emb = self.embedding_net(n20)
        return anchor_emb, n1_emb, n2_emb, n3_emb, n4_emb, n5_emb, n6_emb, n7_emb, n8_emb, n9_emb, n10_emb, n11_emb, n12_emb, n13_emb, n14_emb, n15_emb, n16_emb, n17_emb, n18_emb, n19_emb, n20_emb

    def embed(self, x):
        return self.embedding_net(x)