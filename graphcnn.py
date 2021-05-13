import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("models/")
from mlp import MLP, RES
from torch.nn.parameter import Parameter

class GraphCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers ,dropout, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            dropout: dropout ratio on the final linear layer
            device: which device to use
        '''
        super(GraphCNN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        ### List of MLPs
        self.layers = layers
        self.mlps = nn.ModuleList()
        self.res = nn.ModuleList()

        for layer in range(self.layers):
            if layer == 0:
                self.mlps.append(MLP(input_dim, hidden_dim))
                self.res.append(RES(input_dim, hidden_dim))
            else:
                self.mlps.append(MLP(hidden_dim, hidden_dim))
                self.res.append(RES(hidden_dim, hidden_dim))

        self.dropout = dropout

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix
        start_idx = [0]
        Adj_block = torch.zeros(len(batch_graph)*90,len(batch_graph)*90)

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph))       ## [0,90,180,270,,,]
            Adj_block[start_idx[i]:start_idx[i+1],start_idx[i]:start_idx[i+1]] = batch_graph[i]

        return Adj_block.to(self.device)

    def decoder(self, latent, batch_size):

        num_of_node = [0]
        adj_list = []
        for i in range(batch_size):
            num_of_node.append(num_of_node[i] + 90)  ## number of node ex [0,90,180,270,360,..] [33]
            b_latent = latent[num_of_node[i]:num_of_node[i + 1], 0:self.hidden_dim]  # slicing
            adj = torch.mm(b_latent, b_latent.t()).to(self.device)  ## innerproduct 90x90 ## USE relu
            #adj = F.relu(adj)  ## activation function : ReLU
            adj_list.append(adj)
        adj_con = torch.cat([adjacency for adjacency in adj_list]).to(self.device)  ## 2880x90
        return adj_con

    def encoder(self, graphs):
        feature_matrix = torch.eye(90)  ### create identity matrix
        raw_x = torch.cat([feature_matrix for graph in graphs]).to(self.device)  ## 2880 x 90

        Adj_block = self.__preprocess_neighbors_sumavepool(graphs)  ## 2880 x 2880

        encoding = 1
        ## (0 : GCN / 1 : raw residual / 2 : naive residual / 3 : graph raw residual / 4 : graph naive residual)
        ## (5 : VGAE)

        if encoding == 0:
            ## non residual (gcn)
            x = raw_x
            for i in range(self.layers - 1):
                x = F.relu(self.mlps[i](x, Adj_block))
                x = F.dropout(x, self.dropout, training=self.training)
            Y = F.relu(self.mlps[self.layers - 1](x, Adj_block))

        elif encoding == 1:
            # raw residual

            x = self.mlps[0](raw_x,Adj_block)
            for i in range(1,self.layers - 1):
                x = F.relu(self.mlps[i](x, Adj_block)) + self.res[1](self.mlps[0](raw_x,Adj_block))
                #x = F.dropout(x, self.dropout, training=self.training)

            Y = F.relu(self.mlps[self.layers - 1](x, Adj_block) + self.res[self.layers - 1](self.res[1](self.mlps[0](raw_x,Adj_block))))

        elif encoding == 2:
            # naive residual
            x = raw_x
            for i in range(self.layers - 1):
                x = F.relu(self.mlps[i](x, Adj_block)) + self.res[i](x)
                #x = F.dropout(x, self.dropout, training=self.training)

            Y = F.relu(self.mlps[self.layers - 1](x, Adj_block) + self.res[self.layers - 1](x))

        elif encoding == 3:

            # graph raw residual
            x = raw_x
            for i in range(self.layers - 1):
                x = F.relu(self.mlps[i](x, Adj_block)) + torch.spmm(Adj_block, self.res[0](raw_x))
                x = F.dropout(x, self.dropout, training=self.training)

            Y = F.relu(self.mlps[self.layers - 1](x, Adj_block) + torch.spmm(Adj_block, self.res[self.layers - 1](self.res[0](raw_x))))

        elif encoding == 4:
            #  graph naive residual
            x = raw_x
            for i in range(self.layers - 1):
                x = F.relu(self.mlps[i](x, Adj_block)) + torch.spmm(Adj_block, self.res[i](x))
                x = F.dropout(x, self.dropout, training=self.training)
            Y = F.relu(self.mlps[self.layers - 1](x, Adj_block) + torch.spmm(Adj_block, self.res[self.layers - 1](x)))

        elif encoding == 5:
            x = raw_x
            hidden = F.relu(self.mlps[0](x,Adj_block))
            self.mean = self.mlps[1](hidden,Adj_block)
            self.logstd = self.mlps[2](hidden,Adj_block)
            gaussian_noise = torch.randn(x.size(0), self.hidden_dim).to(self.device)
            Y = F.relu(gaussian_noise * torch.exp(self.logstd) + self.mean)

        return Y

    def forward(self, graphs):

        Y = self.encoder(graphs)
        Z = self.decoder(Y, len(graphs))

        return Z
