import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self,input_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = Parameter(torch.FloatTensor(output_dim))
        # self.eps = nn.Parameter(torch.zeros(1)) ## self learning
        ## weight initialization
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        #nn.init.normal_(self.weight.data, mean=0.05, std=0.01)


        nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) + self.bias #

        return output

class RES(nn.Module):
    def __init__(self,input_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(RES, self).__init__()

        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = Parameter(torch.FloatTensor(output_dim))

        ## weight initialization
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        #nn.init.normal_(self.weight.data, mean=0.05, std=0.01)
        nn.init.zeros_(self.bias.data)

    def forward(self, input):

        output = torch.mm(input, self.weight) + self.bias
        return output