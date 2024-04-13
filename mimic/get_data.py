"""
From MultiBench: https://github.com/pliang279/MultiBench/blob/main/datasets/mimic/get_data.py
"""

"""Implements dataloaders for generic MIMIC tasks."""
import numpy as np
from tqdm import tqdm
import sys
import os
import numpy as np
import random
import pickle
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

"""
NOTE: use task = -1 for the 6-way mortality classification task. 

Mortality classes are predicting on how long from the date the data was recorded that the patient would die. This includes: 
- 1 day
- 2 day
- 3 day
- 1 week
- 1 year
- 1+ year
"""
def get_data(task, imputed_path='data/im.pk'):
    """Get datasets for MIMIC 

    Args:
        task (int): Integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task.
        imputed_path (str, optional): Datafile location. Defaults to 'im.pk'.
    Returns:
        tuple: Tuple of training dataloader, validation dataloader, and test dataloader
    """
    f = open(imputed_path, 'rb')
    datafile = pickle.load(f)
    f.close()
    X_t = datafile['ep_tdata']
    X_s = datafile['adm_features_all']

    X_t[np.isinf(X_t)] = 0
    X_t[np.isnan(X_t)] = 0
    X_s[np.isinf(X_s)] = 0
    X_s[np.isnan(X_s)] = 0

    X_s_avg = np.average(X_s, axis=0)
    X_s_std = np.std(X_s, axis=0)
    X_t_avg = np.average(X_t, axis=(0, 1))
    X_t_std = np.std(X_t, axis=(0, 1))

    for i in range(len(X_s)):
        X_s[i] = (X_s[i]-X_s_avg)/X_s_std
        for j in range(len(X_t[0])):
            X_t[i][j] = (X_t[i][j]-X_t_avg)/X_t_std

    static_dim = len(X_s[0])
    timestep = len(X_t[0])
    series_dim = len(X_t[0][0])
    if task < 0:
        y = datafile['adm_labels_all'][:, 1]
        admlbl = datafile['adm_labels_all']
        le = len(y)
        for i in range(0, le):
            if admlbl[i][1] > 0:
                y[i] = 1
            elif admlbl[i][2] > 0:
                y[i] = 2
            elif admlbl[i][3] > 0:
                y[i] = 3
            elif admlbl[i][4] > 0:
                y[i] = 4
            elif admlbl[i][5] > 0:
                y[i] = 5
            else:
                y[i] = 0
    else:
        y = datafile['y_icd9'][:, task]
        le = len(y)
    datasets = [(X_s[i], X_t[i], y[i]) for i in range(le)]

    random.shuffle(datasets)

    trains = datasets[le//5:]

    valids = datasets[0:le//10]
    
    tests = datasets[le//10:le//5]


    return trains, valids, tests

def train_sampler(train_dataset):
    # extract label from dataset
    labels = [sample[2] for sample in train_dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
    return sampler

if __name__ == "__main__":
    dirpath = "../data/mimic/im.pk"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setattr(args, 'data_path', dirpath)

    train_set, val_set, test_set = get_data(-1, imputed_path=dirpath)

    #train_sampler = make_balanced_sampler(train_set.label)

    # train_loader = DataLoader(
    #     train_set, 
    #     batch_size=16, 
    #     collate_fn=train_set.custom_collate, 
    #     sampler=train_sampler
    # )