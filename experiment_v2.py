from collections import Counter
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from utils import TripletLossWeighted, TripletLoss, TripletDataset, calc_embeddings
import pandas as pd
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
from itertools import combinations
from torchvision import transforms

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
                     'lr': 0.0002,
                     'batch_size': 16,
                     'gamma': 0.99,
                     'epochs': 55}

    config['dermatology'] = {'nn_config': {'units_1st_layer': 97,
                                           'units_2nd_layer': 512,
                                           'units_3rd_layer': 256,
                                           'units_latent_layer': 16},
                             'weighted_triplet_loss': True,
                             'lr': 0.0010,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 45}

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
                             'lr': 0.001,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 150}

    config['new_yeast'] = {'nn_config': {'units_1st_layer': 9,
                                         'units_2nd_layer': 300,
                                         'units_3rd_layer': 200,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0004,
                           'batch_size': 32,
                           'gamma': 0.99,
                           'epochs': 60}

    config['balance-scale'] = {'nn_config': {'units_1st_layer': 16,
                                             'units_2nd_layer': 256,
                                             'units_3rd_layer': 128,
                                             'units_latent_layer': 10},
                               'weighted_triplet_loss': True,
                               'lr': 0.007,
                               'batch_size': 16,
                               'gamma': 0.99,
                               'epochs': 100}

    config['cleveland'] = {'nn_config': {'units_1st_layer': 24,
                                         'units_2nd_layer': 72,
                                         'units_3rd_layer': 48,
                                         'units_latent_layer': 16},
                           'weighted_triplet_loss': True,
                           'lr': 0.00002,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 120}

    config['cleveland_v2'] = {'nn_config': {'units_1st_layer': 23,
                                            'units_2nd_layer': 256,
                                            'units_3rd_layer': 128,
                                            'units_latent_layer': 16},
                              'weighted_triplet_loss': True,
                              'lr': 0.00002,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 100}

    config['glass'] = {'nn_config': {'units_1st_layer': 9,
                                     'units_2nd_layer': 256,
                                     'units_3rd_layer': 128,
                                     'units_latent_layer': 12},
                       'weighted_triplet_loss': True,
                       'lr': 0.001,
                       'batch_size': 16,
                       'gamma': 0.99,
                       'epochs': 120}

    config['thyroid-newthyroid'] = {'nn_config': {'units_1st_layer': 5,
                                                  'units_2nd_layer': 64,
                                                  'units_3rd_layer': 32,
                                                  'units_latent_layer': 8},
                                    'weighted_triplet_loss': True,
                                    'lr': 0.001,
                                    'batch_size': 16,
                                    'gamma': 0.99,
                                    'epochs': 120}

    config['new_ecoli'] = {'nn_config': {'units_1st_layer': 7,
                                         'units_2nd_layer': 128,
                                         'units_3rd_layer': 64,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0001,
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
                                        'lr': 0.004,
                                        'batch_size': 16,
                                        'gamma': 0.99,
                                        'epochs': 120}

    config['3mocniej-cut'] = {'nn_config': {'units_1st_layer': 2,
                                            'units_2nd_layer': 128,
                                            'units_3rd_layer': 64,
                                            'units_latent_layer': 10},
                              'weighted_triplet_loss': True,
                              'lr': 0.04,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 90}

    config['1czysty-cut'] = {'nn_config': {'units_1st_layer': 2,
                                           'units_2nd_layer': 64,
                                           'units_3rd_layer': 32,
                                           'units_latent_layer': 8},
                             'weighted_triplet_loss': True,
                             'lr': 0.004,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 130}

    config['2delikatne-cut'] = {'nn_config': {'units_1st_layer': 2,
                                              'units_2nd_layer': 128,
                                              'units_3rd_layer': 64,
                                              'units_latent_layer': 12},
                                'weighted_triplet_loss': True,
                                'lr': 0.04,
                                'batch_size': 16,
                                'gamma': 0.99,
                                'epochs': 90}
    return config


def config_tuned_for_dt(config):
    config['cmc'] = {'nn_config': {'units_1st_layer': 17,
                                   'units_2nd_layer': 256,
                                   'units_3rd_layer': 128,
                                   'units_latent_layer': 8},
                     'weighted_triplet_loss': True,
                     'lr': 0.0002,
                     'batch_size': 16,
                     'gamma': 0.99,
                     'epochs': 55}

    config['dermatology'] = {'nn_config': {'units_1st_layer': 97,
                                           'units_2nd_layer': 512,
                                           'units_3rd_layer': 256,
                                           'units_latent_layer': 8},
                             'weighted_triplet_loss': True,
                             'lr': 0.0010,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 70}

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
                             'lr': 0.001,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 150}

    config['new_yeast'] = {'nn_config': {'units_1st_layer': 9,
                                         'units_2nd_layer': 300,
                                         'units_3rd_layer': 200,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0004,
                           'batch_size': 32,
                           'gamma': 0.99,
                           'epochs': 60}

    config['balance-scale'] = {'nn_config': {'units_1st_layer': 16,
                                             'units_2nd_layer': 256,
                                             'units_3rd_layer': 128,
                                             'units_latent_layer': 10},
                               'weighted_triplet_loss': True,
                               'lr': 0.007,
                               'batch_size': 16,
                               'gamma': 0.99,
                               'epochs': 100}

    config['cleveland'] = {'nn_config': {'units_1st_layer': 24,
                                         'units_2nd_layer': 72,
                                         'units_3rd_layer': 48,
                                         'units_latent_layer': 8},
                           'weighted_triplet_loss': True,
                           'lr': 0.00002,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 120}

    config['cleveland_v2'] = {'nn_config': {'units_1st_layer': 23,
                                            'units_2nd_layer': 256,
                                            'units_3rd_layer': 128,
                                            'units_latent_layer': 16},
                              'weighted_triplet_loss': True,
                              'lr': 0.00002,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 100}

    config['glass'] = {'nn_config': {'units_1st_layer': 9,
                                     'units_2nd_layer': 256,
                                     'units_3rd_layer': 128,
                                     'units_latent_layer': 12},
                       'weighted_triplet_loss': True,
                       'lr': 0.001,
                       'batch_size': 16,
                       'gamma': 0.99,
                       'epochs': 120}

    config['thyroid-newthyroid'] = {'nn_config': {'units_1st_layer': 5,
                                                  'units_2nd_layer': 64,
                                                  'units_3rd_layer': 32,
                                                  'units_latent_layer': 8},
                                    'weighted_triplet_loss': True,
                                    'lr': 0.001,
                                    'batch_size': 16,
                                    'gamma': 0.99,
                                    'epochs': 120}

    config['new_ecoli'] = {'nn_config': {'units_1st_layer': 7,
                                         'units_2nd_layer': 128,
                                         'units_3rd_layer': 64,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0001,
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
                                                   'units_latent_layer': 8},
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
                                        'lr': 0.004,
                                        'batch_size': 16,
                                        'gamma': 0.99,
                                        'epochs': 120}

    config['3mocniej-cut'] = {'nn_config': {'units_1st_layer': 2,
                                            'units_2nd_layer': 128,
                                            'units_3rd_layer': 64,
                                            'units_latent_layer': 4},
                              'weighted_triplet_loss': True,
                              'lr': 0.04,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 90}

    config['1czysty-cut'] = {'nn_config': {'units_1st_layer': 2,
                                           'units_2nd_layer': 64,
                                           'units_3rd_layer': 32,
                                           'units_latent_layer': 4},
                             'weighted_triplet_loss': True,
                             'lr': 0.004,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 130}

    config['2delikatne-cut'] = {'nn_config': {'units_1st_layer': 2,
                                              'units_2nd_layer': 128,
                                              'units_3rd_layer': 64,
                                              'units_latent_layer': 12},
                                'weighted_triplet_loss': True,
                                'lr': 0.04,
                                'batch_size': 16,
                                'gamma': 0.99,
                                'epochs': 90}
    return config

def config_tuned_for_knn(config):
    config['cmc'] = {'nn_config': {'units_1st_layer': 17,
                                   'units_2nd_layer': 256,
                                   'units_3rd_layer': 128,
                                   'units_latent_layer': 8},
                     'weighted_triplet_loss': True,
                     'lr': 0.0002,
                     'batch_size': 16,
                     'gamma': 0.99,
                     'epochs': 55}

    config['dermatology'] = {'nn_config': {'units_1st_layer': 97,
                                           'units_2nd_layer': 512,
                                           'units_3rd_layer': 256,
                                           'units_latent_layer': 8},
                             'weighted_triplet_loss': True,
                             'lr': 0.0010,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 45}

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
                             'lr': 0.001,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 150}

    config['new_yeast'] = {'nn_config': {'units_1st_layer': 9,
                                         'units_2nd_layer': 300,
                                         'units_3rd_layer': 200,
                                         'units_latent_layer': 12},
                           'weighted_triplet_loss': True,
                           'lr': 0.0004,
                           'batch_size': 32,
                           'gamma': 0.99,
                           'epochs': 60}

    config['balance-scale'] = {'nn_config': {'units_1st_layer': 16,
                                             'units_2nd_layer': 256,
                                             'units_3rd_layer': 128,
                                             'units_latent_layer': 10},
                               'weighted_triplet_loss': True,
                               'lr': 0.007,
                               'batch_size': 16,
                               'gamma': 0.99,
                               'epochs': 100}

    config['cleveland'] = {'nn_config': {'units_1st_layer': 24,
                                         'units_2nd_layer': 72,
                                         'units_3rd_layer': 48,
                                         'units_latent_layer': 16},
                           'weighted_triplet_loss': True,
                           'lr': 0.00002,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 120}

    config['cleveland_v2'] = {'nn_config': {'units_1st_layer': 23,
                                            'units_2nd_layer': 128,
                                            'units_3rd_layer': 64,
                                            'units_latent_layer': 16},
                              'weighted_triplet_loss': True,
                              'lr': 0.00002,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 100}

    config['glass'] = {'nn_config': {'units_1st_layer': 9,
                                     'units_2nd_layer': 256,
                                     'units_3rd_layer': 128,
                                     'units_latent_layer': 12},
                       'weighted_triplet_loss': True,
                       'lr': 0.001,
                       'batch_size': 16,
                       'gamma': 0.99,
                       'epochs': 120}

    config['thyroid-newthyroid'] = {'nn_config': {'units_1st_layer': 5,
                                                  'units_2nd_layer': 64,
                                                  'units_3rd_layer': 32,
                                                  'units_latent_layer': 8},
                                    'weighted_triplet_loss': True,
                                    'lr': 0.001,
                                    'batch_size': 16,
                                    'gamma': 0.99,
                                    'epochs': 120}

    config['new_ecoli'] = {'nn_config': {'units_1st_layer': 7,
                                         'units_2nd_layer': 128,
                                         'units_3rd_layer': 64,
                                         'units_latent_layer': 8},
                           'weighted_triplet_loss': True,
                           'lr': 0.0001,
                           'batch_size': 16,
                           'gamma': 0.99,
                           'epochs': 100}

    config['new_led7digit'] = {'nn_config': {'units_1st_layer': 7,
                                             'units_2nd_layer': 64,
                                             'units_3rd_layer': 32,
                                             'units_latent_layer': 8},
                               'weighted_triplet_loss': True,
                               'lr': 0.003,
                               'batch_size': 16,
                               'gamma': 0.99,
                               'epochs': 100}

    config['new_winequality-red'] = {'nn_config': {'units_1st_layer': 11,
                                                   'units_2nd_layer': 128,
                                                   'units_3rd_layer': 64,
                                                   'units_latent_layer': 8},
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
                                        'lr': 0.004,
                                        'batch_size': 16,
                                        'gamma': 0.99,
                                        'epochs': 120}

    config['3mocniej-cut'] = {'nn_config': {'units_1st_layer': 2,
                                            'units_2nd_layer': 128,
                                            'units_3rd_layer': 64,
                                            'units_latent_layer': 4},
                              'weighted_triplet_loss': True,
                              'lr': 0.04,
                              'batch_size': 16,
                              'gamma': 0.99,
                              'epochs': 90}

    config['1czysty-cut'] = {'nn_config': {'units_1st_layer': 2,
                                           'units_2nd_layer': 64,
                                           'units_3rd_layer': 32,
                                           'units_latent_layer': 4},
                             'weighted_triplet_loss': True,
                             'lr': 0.004,
                             'batch_size': 16,
                             'gamma': 0.99,
                             'epochs': 130}

    config['2delikatne-cut'] = {'nn_config': {'units_1st_layer': 2,
                                              'units_2nd_layer': 128,
                                              'units_3rd_layer': 64,
                                              'units_latent_layer': 12},
                                'weighted_triplet_loss': True,
                                'lr': 0.04,
                                'batch_size': 16,
                                'gamma': 0.99,
                                'epochs': 90}
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


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, nn_config):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = nn_config["units_decision_layer"]
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(nn_config["units_latent_layer"], nn_config["units_decision_layer"])

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def embed(self, x):
        return self.embedding_net(x)


def train_classification_net(model, device, train_loader, optimizer):
    model.train()
    train_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)


def test_classification_net(model, device, test_loader):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target.long())

            test_loss.append(loss.item())
    return np.mean(test_loss)


def train_tripletnet(model, device, train_loader, optimizer, epoch, weights, nn_config, log_interval=10, dry_run=False):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = (data,)
        data = tuple(d.cuda() for d in data)

        target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)
        outputs = (outputs,)
        loss_inputs = outputs

        target = (target,)
        loss_inputs += target

        margin = 1.0
        loss_fn = OnlineTripletLoss(margin=margin, triplet_selector=RandomNegativeTripletSelector(margin))

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


def test_tripletnet(model, device, test_loader, weights, nn_config):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # print(data)
            # print(target)
            data = (data,)
            data = tuple(d.cuda() for d in data)

            target = target.cuda()

            outputs = model(*data)
            outputs = (outputs,)

            loss_inputs = outputs

            target = (target,)
            loss_inputs += target

            margin = 1.0
            loss_fn = OnlineTripletLoss(margin=margin, triplet_selector=RandomNegativeTripletSelector(margin))

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            test_loss.append(loss.item())
    # print(test_loss)
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

    train_kwargs, test_kwargs = {}, {}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
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

    triplet_train_dataset = TripletDataset(dataset1)
    triplet_test_dataset = TripletDataset(dataset2)

    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, **train_kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, **test_kwargs)

    n_classes = np.unique(y_train).size

    train_batch_sampler = BalancedBatchSampler(dataset1.train_labels, n_classes=n_classes, n_samples=max(1,batch_size//n_classes), name="train")
    test_batch_sampler = BalancedBatchSampler(dataset2.test_labels, n_classes=n_classes, n_samples=max(1,test_batch_size//n_classes), name="test")

    online_train_loader = torch.utils.data.DataLoader(dataset1, batch_sampler=train_batch_sampler, **train_kwargs)
    online_test_loader = torch.utils.data.DataLoader(dataset2, batch_sampler=test_batch_sampler, **test_kwargs)

    embedding_net = EmbeddingNet(nn_config)
    # model = TripletNet(embedding_net).to(device)
    model = embedding_net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    test_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        train_losses.append(train_tripletnet(model, device, online_train_loader, optimizer, epoch, weights, nn_config, log_interval))
        test_losses.append(test_tripletnet(model, device, online_test_loader, weights, nn_config))
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


def train_classification(X_train, y_train, X_test, y_test, cfg):
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

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    embedding_net = EmbeddingNet(nn_config)
    model = ClassificationNet(embedding_net, nn_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    test_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        train_losses.append(train_classification_net(model, device, train_loader, optimizer))
        test_losses.append(test_classification_net(model, device, test_loader))
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    plt.plot(test_losses, label="test losses")
    plt.plot(train_losses, label="train losses")
    plt.legend()
    plt.show()
    return model


class BalancedBatchSampler(BatchSampler):
    """
    samples n_classes and within these classes samples n_samples
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels, n_classes, n_samples, name):
        self.name=name
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        # print(f"{self.name}: {self.n_dataset}")
        # print(f"{self.name}: {self.n_classes} x {self.n_samples}")
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # print(self.name)
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


class AverageNonzeroTripletsMetric:
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'