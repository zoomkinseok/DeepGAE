import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def load_data(sparsity, dataset, clipping = False):
    fcs = os.listdir('./dataset/%s' %dataset)
    HCP_adj_list = []
    numnode = 90
    thr = 100 - sparsity

    for fc in fcs:
        adj = np.load('./dataset/%s/%s' %(dataset,fc))

        con = adj[0:numnode, 0:numnode]
        con[con <= 0] = 0
        threshold = np.percentile(con,thr)
        con[con <= threshold] = 0  ### sparsity threshold processing


        if clipping:
            u = np.mean(con)
            o = np.std(con)
            con[con > (2*o + u)] = 2*o + u

        np.fill_diagonal(con, 1)
        HCP_adj_list.append(con)

    return HCP_adj_list

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [0 for i in range(len(graph_list))]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)

    train_idx, test_idx = idx_list[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]


    return train_graph_list, test_graph_list

def normalization(graphs):
    normal_batch = []

    for A_hat in graphs:

        D_hat_diag = np.sum(A_hat, axis=1)
        D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
        D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
        D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
        B = np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        B = torch.from_numpy(B)
        normal_batch.append(B)

    return normal_batch

