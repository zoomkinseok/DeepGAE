import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from utils import load_data, separate_data, normalization
from save import save_feature, save_loss
from graphcnn import GraphCNN
import os

criterion = nn.MSELoss() ## joo
def input_processing(graph_list, device):

    input_data = torch.cat([graph for graph in graph_list]).to(device)  ## 2880 x 90 (tensor)
    return input_data

def train(args, model, device, train_graphs, optimizer, epoch):


    model.train()
    total_iters = args.iters_per_epoch    ## 50
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0

    for pos in pbar:   ## number of iteration : 50 ##stochastic

        ## batch training
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]  ##  [ 32 graphs 's informaion ]

        Z = model(batch_graph)

        input_data = input_processing(batch_graph,device)

        ## nonnegativity constraints
        nonneg_tensor = model.mlps[0].weight.data.clone().detach()
        nonneg_tensor[nonneg_tensor > 0] = 0

        loss = criterion(input_data, Z)
        ## VGAE
        # kl_divergence = 0.5 / Z.size(0) * (
        #             1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()


        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.nonneg == 'True':
                print("apply nonnegativity")
                model.gc1.weight.data = model.gc1.weight.data - (1.4)

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters

    return average_loss


def test(model, device, test_graphs): ## no batch
    model.eval()

    Z = model(test_graphs)

    #output = InnerProduct(Y, size, args.hidden_dim, device)
    input = input_processing(test_graphs, device)

    # compute loss (joo)
    loss = criterion(input,Z)
    loss = loss.detach().cpu().numpy()

    return loss


def main():
    save = True

    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph auto-encoder for decompose whole brain network')
    parser.add_argument('--no', type=str, default=1,
                        help='index of experiment')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--sparsity', type=int, default=20, ## joo
                        help='sparsity of dataset (default: 20)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--seedforw', type=int, default='0',
                        help='seed for changing weight initial')
    parser.add_argument('--nonneg', type=str, default="False",
                        help='Whether to apply non-negativity weight constraints')
    args = parser.parse_args()

    # set up seeds and gpu device
    torch.manual_seed(args.seedforw)
    np.random.seed(args.seedforw)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ## Road data & Separate data
    pre_graphs = load_data(args.sparsity, args.dataset)
    graphs = normalization(pre_graphs) ## DAD
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    ## Define model
    model = GraphCNN(90, args.hidden_dim,args.num_layers, args.final_dropout, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) ##learning late decay

    ## Train & Test
    train_loss_list = []
    validation_loss_list = []
    for epoch in range(1, args.epochs + 1):

        # output loss
        train_loss = train(args, model, device, train_graphs, optimizer, epoch)
        train_loss_list.append(train_loss)
        print("loss training: %f" % (train_loss))
        scheduler.step()

        with torch.no_grad():
            test_loss = test(model, device, test_graphs)
            validation_loss_list.append(test_loss)
            print("test loss : ", test_loss)

    ## Save feature,loss,weight
    if save:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir,
                                   '%s__sparsity:%d__hidden_dim:%d__num_layer:%d__nonneg:%s__learning_rate:%f' % (
                                       args.no, args.sparsity, args.hidden_dim, args.num_layers, args.nonneg, args.lr))
        fold_dir = os.path.join(results_dir, 'FOLD : %d' % args.fold_idx)

        if not os.path.isdir(fold_dir):
            os.makedirs(fold_dir)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        save_feature(args,model, graphs, fold_dir)
        save_loss(args,train_loss_list,validation_loss_list,fold_dir)
        torch.save(model.state_dict(),
                   fold_dir + '/' + 'Weight__sparsity:%d__hidden_dim:%d__num_layer:%d__nonneg:%s__learning_rate:%f__FOLD_idx:%d.pt' % (
                       args.sparsity, args.hidden_dim, args.num_layers, args.nonneg, args.lr, args.fold_idx))


if __name__ == '__main__':
    main()