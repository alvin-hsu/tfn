import numpy
import torch
import torch.nn
import torch.nn.init as init
import torch.nn.functional as F

from .utils import *

class RotEquiv_Nonlin(nn.Module):
    def __init__(self, channels, nonlin=ssp, b_init=None):
        if b_init is None:
            b_init = init.zeros_

        self.channels = channels
        self.nonlin = nonlin

        self.bias = nn.Parameter(torch.tensor(channels))

        self.register_parameter('bias', self.biases)

        b_init(self.bias)

    def forward(x):
        channels = x.size(-2)
        rep_idx = x.size(-1)

        assert(channels == self.channels)

        if rep_idx == 1:
            return self.nonlin(x)

        else:
            norm = eps_l2_norm(x, axis=-1)
            nonlin_out = self.nonlin(norm + self.bias)
            factor = nonlin_out / norm
            return x * torch.unsqueeze(factor, -1)


class RadialConv(nn.Module):
    def __init__(self, n_input, n_hidden=None, n_output=1, nonlin=F.relu,
                 W_init=None, b_init=None):
        if self.n_hidden is None:
            self.n_hidden = n_input

        if W_init is None:
            W_init = init.xavier_normal_
        if b_init is None:
            b_init = init.zeros_

        self.nonlin = nonlin

        self.weight1 = nn.Parameter(torch.tensor(n_hidden, n_input))
        self.weight2 = nn.Parameter(torch.tensor(n_output, n_hidden))
        self.bias1 = nn.Parameter(torch.tensor(n_hidden))
        self.bias2 = nn.Parameter(torch.tensor(n_output))

        self.register_parameter('weight1', self.weight1)
        self.register_parameter('weight2', self.weight2)
        self.register_parameter('bias1', self.bias1)
        self.register_parameter('bias2', self.bias2)

        W_init(self.weight1)
        W_init(self.weight2)
        b_init(self.bias1)
        b_init(self.bias2)

    def forward(self, x):
        x = self.bias1 + torch.einsum('ijk,lk->ijl', x, self.weight1)
        x = self.nonlin(x)
        return self.bias2 + torch.einsum('ijk,lk->ijl', x, self.weight2)
        



