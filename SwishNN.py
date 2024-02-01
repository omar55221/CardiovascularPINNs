import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import *


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class MySquared(nn.Module):
    def __init__(self, inplace=True):
        super(MySquared, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.square(x)


class SwishNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers):
        super(SwishNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim_in, dim_hidden))
        layers.append(Swish())
       # layers.append(nn.Linear(dim_hidden, dim_hidden))
        #layers.append(Swish())
        for i in range(num_layers-1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(Swish())
        layers.append(nn.Linear(dim_hidden, dim_out))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        output = self.layers(x)
        return output
