import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import *


class Tanh(nn.Module):
    def __init__(self, inplace=True):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return torch.tanh(x)
        else:
            return torch.tanh(x)


class MySquared(nn.Module):
    def __init__(self, inplace=True):
        super(MySquared, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.square(x)


class TanhNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers):
        super(TanhNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim_in, dim_hidden))
        layers.append(Tanh())
       # layers.append(nn.Linear(dim_hidden, dim_hidden))
        #layers.append(Swish())
        for i in range(num_layers-1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(Tanh())
        layers.append(nn.Linear(dim_hidden, dim_out))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        output = self.layers(x)
        return output
