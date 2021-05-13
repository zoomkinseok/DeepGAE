import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def degree(graphs):
    degree_graphs = []
    for i, A_hat in enumerate(graphs):
        D_hat_diag = torch.sum(A_hat, axis=1)
        D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, 0.5)
        D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0.
        D_hat_inv_sqrt = torch.diag(D_hat_diag_inv_sqrt)
        degree_graphs.append(D_hat_inv_sqrt)

    return degree_graphs


def extract_deep_feature(model, graphs):
    model.eval()
    batch_size = 108
    output = []
    denormal_list = []
    idx = np.arange(len(graphs))  ##[0,1,2,...1079]
    degree_graphs = degree(graphs)

    for i in range(0, len(graphs), batch_size):
        sampled_idx = idx[i:i + batch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model.encoder([graphs[j] for j in
                                     sampled_idx]).detach().cpu())  ## model(minibatch graph #108) , output = [(64x90 x 90),(64x90 x 90),(64x90 x 90)...]

    deep_feature = torch.cat(output, 0)  ##deep_list  ##return = ((Number of test_graph)x90) x 90
    deep_list = torch.chunk(deep_feature, len(graphs), dim=0)

    # for i in range(len(graphs)):
    #
    #     denormal_list.append(torch.mm(degree_graphs[i],deep_list[i]))

    return deep_list


def save_feature(args, model, graphs, fold_dir):
    npy_file_name = 'Deep_feature__sparsity:%d__hidden_dim:%d__num_layer:%d__nonneg:%s__learning_rate:%f__FOLD_idx:%d.npy' % (
        args.sparsity, args.hidden_dim, args.num_layers, args.nonneg, args.lr, args.fold_idx)

    deep_feature = extract_deep_feature(model, graphs)
    deep_feature_tensor = deep_feature[0].cpu().numpy()
    deep_feature_tensor = deep_feature_tensor.reshape(1, 90, -1)

    for i in range(1, len(graphs)):
        deep_feature_tensor = np.append(deep_feature_tensor, deep_feature[i].cpu().numpy().reshape(1, 90, -1),
                                        axis=0)

    one_deep = deep_feature_tensor[500]
    mid_deep = np.median(deep_feature_tensor, axis=0)
    avg_deep = np.mean(deep_feature_tensor, axis=0)
    std_deep = np.std(deep_feature_tensor, axis=0)

    np.save(fold_dir + '/One_' + npy_file_name, one_deep)
    np.save(fold_dir + '/Avg_' + npy_file_name, avg_deep)
    np.save(fold_dir + '/Std_' + npy_file_name, std_deep)
    np.save(fold_dir + '/Mid_' + npy_file_name, mid_deep)


def save_loss(args, train_loss_list, validation_loss_list, fold_dir):
    ## save loss and acc csv
    csv_list = np.column_stack([train_loss_list, validation_loss_list])
    csv_file_name = 'Loss_CSV__sparsity:%d__hidden_dim:%d__num_layer:%d__nonneg:%s__learning_rate:%f__FOLD_idx:%d.csv' % (
        args.sparsity, args.hidden_dim, args.num_layers, args.nonneg, args.lr, args.fold_idx)

    pd.DataFrame(csv_list, columns=['train loss', 'validation loss']).to_csv(fold_dir + '/' + csv_file_name)
    plt.subplot(3, 1, 1)
    plt.plot(validation_loss_list, 'g-')
    plt.plot(train_loss_list, 'r-')
    plt.legend(['Validation loss', 'Train loss'])
    plt.title('loss graph')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(validation_loss_list, 'g-')
    plt.title('Validation loss curve')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(train_loss_list, 'r-')
    plt.title('Train loss curve')
    plt.grid(True)

    plt.savefig(
        fold_dir + '/' + 'Loss curve__sparsity:%d__hidden_dim:%d__num_layer:%d__nonneg:%s__learning_rate:%f__FOLD_idx:%d.png' % (
            args.sparsity, args.hidden_dim, args.num_layers, args.nonneg, args.lr, args.fold_idx))
    plt.close()

