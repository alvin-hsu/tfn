import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .utils import *


class Radial(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=None, nonlin=F.relu,
                 W_init=None, b_init=None):
        super(Radial, self).__init__()

        if n_hidden is None:
            n_hidden = n_input
        if W_init is None:
            W_init = init.xavier_normal_
        if b_init is None:
            b_init = init.zeros_

        self.nonlin = nonlin

        self.weight1 = nn.Parameter(torch.Tensor(n_hidden, n_input))
        self.weight2 = nn.Parameter(torch.Tensor(n_output, n_hidden))
        self.bias1 = nn.Parameter(torch.Tensor(n_hidden))
        self.bias2 = nn.Parameter(torch.Tensor(n_output))

        self.register_parameter('weight1', self.weight1)
        self.register_parameter('weight2', self.weight2)
        self.register_parameter('bias1', self.bias1)
        self.register_parameter('bias2', self.bias2)

        W_init(self.weight1)
        W_init(self.weight2)
        b_init(self.bias1)
        b_init(self.bias2)

    def forward(self, rbf):
        # [batch, pts, pts, rbf_ctrs] -> [batch, pts, pts, channels]
        hidden = self.bias1 + torch.einsum('ijkl,ml->ijkm', rbf, self.weight1)
        hidden = self.nonlin(hidden)
        return self.bias2 + torch.einsum('ijkl,ml->ijkm', hidden, self.weight2)


class Harmonic(nn.Module):
    def __init__(self, order):
        super(Harmonic, self).__init__()

        self.order = order

        if self.order < 0:
            raise ValueError('Spherical harmonic must be of positive order!')
        if self.order > 2:
            raise NotImplementedError('Spherical harmonics only supported up to order 2.')

    def forward(self, rij):
        if self.order == 0:
            return torch.ones(rij.size(0), rij.size(1), rij.size(2), 1)
        elif self.order == 1:
            return rij / eps_l2_norm(rij, dim=-1, keepdim=True)
        elif self.order == 2:
            x = rij[:, :, :, 0]  # [batch, pts, pts]
            y = rij[:, :, :, 1]  # [batch, pts, pts]
            z = rij[:, :, :, 2]  # [batch, pts, pts]
            r2 = torch.max(torch.sum(rij**2, dim=-1), EPS)
            output = torch.stack([
                    x * y / r2,
                    y * z / r2,
                    (2 * z**2 - x**2 - y**2) / (2 * np.sqrt(3) * r2),
                    z * x / r2,
                    (x**2 - y**2) / (2 * r2)
                ], dim=-1)  # [batch, pts, pts, 5]
            return output
        else:
            raise NotImplementedError('Spherical harmonics only supported up to order 2.')


class Filter(nn.Module):
    Harmonics = [Harmonic(0), Harmonic(1), Harmonic(2)]
    def __init__(self, rbf_size, n_chan, rank_in, rank_out, order):
        super(Filter, self).__init__()
        self.order = order

        self.radial = Radial(rbf_size, n_chan)
        self.Y = Filter.Harmonics[order]
        
        # Transformation Law  FIXME: make correct/not hacky
        try:
            self.cg = torch.eye(max(2*rank_out + 1, 2*order + 1))
            self.cg = self.cg.view(2*rank_out + 1, 2*order + 1, 2*rank_in + 1)
        except RuntimeError:
            self.cg = EIJK  # Levi-Civita Tensor

    def forward(self, tensor, rbf, rij):
        # Radial component
        R = self.radial(rbf)  # [batch, pts, pts, n_chan]
        if self.order != 0:  # Mask for order > 1
            mask = (torch.sqrt(torch.sum(rij**2, dim=-1, keepdim=True)) < EPS)  # [batch, pts, pts, 1]
            mask = torch.cat(R.size(-1) * [mask], dim=-1)  # [batch, pts, pts, n_chan]
            R = torch.where(mask, torch.zeros_like(R), R)  # [batch, pts, pts, n_chan]
        # Angular component
        Y = self.Y(rij)  # [batch, pts, pts, f_rep]
        # Combined filter
        F_out = torch.unsqueeze(R, -1) * torch.unsqueeze(Y, -2)  # [batch, pts, pts, n_chan, f_rep]

        # Transformation law + apply filter
        # [x, y, z] [batch, p, p, n_chan, f_rep] [batch, p, n_chan, xyz] -> [batch, p, n_chan, xyz]
        return torch.einsum('ijk,abcdj,acdk->abdi', self.cg, F_out, tensor)


class Convolution(nn.Module):
    def __init__(self, rbf_size, channels, ranks_in, ranks_out, filters=None):
        super(Convolution, self).__init__()
        # Dict with keys for each entry in ranks_in
        self.channels = channels
        # Lists
        self.ranks_in = ranks_in
        self.ranks_out = ranks_out
        # Either specify each filter by hand or construct them all.
        if filters is None:
            filters = {}
            for rank_in in ranks_in:
                filters[rank_in] = {}
                for rank_out in ranks_out:
                    filters[rank_in][rank_out] = {}
                    for order in range(rank_out):
                        filters[rank_in][rank_out][order] = Filter(
                                rbf_size, channels[rank_in], rank_in, rank_out, order)

        self.filters = filters  # 3D dict of Filters: self.filters[rank_in][rank_out][order]

    def forward(self, in_dict, rbf, rij):
        out_lists = { k : [] for k in self.ranks_out }
        for rank_in in self.ranks_in:
            for rank_out in self.ranks_out:
                for f in self.filters[rank_in][rank_out].values():
                    out_lists[rank_out].append(f(in_dict[rank_in], rbf, rij))
        
        # Each list in out_lists contains tensors of shape [points, channels, rep_dims]
        out_dict = { k : torch.cat(v, -2) for k, v in out_lists.items() }
        return out_dict, rbf, rij


class SelfInteraction(nn.Module):
    def __init__(self, channels_in, channels_out, W_init=None, b_init=None, bias=True):
        super(SelfInteraction, self).__init__()
        if W_init is None:
            W_init = init.xavier_normal_
        if b_init is None:
            b_init = init.zeros_

        # channels_in and channels_out are dicts { rank : channels }
        self.ranks = list(channels_in.keys())
        self.weights = {}
        for rank in channels_in.keys():
            self.weights[rank] = nn.Parameter(
                    torch.Tensor(channels_out[rank], channels_in[rank]))
            self.register_parameter('W_%i' % rank, self.weights[rank])
            W_init(self.weights[rank])

        try:
            assert bias is True
            self.bias = nn.Parameter(torch.Tensor(channels_out[0], 1))
            self.register_parameter('b', self.bias)
            b_init(self.bias)
        except AssertionError:
            self.bias = None
        except KeyError:
            self.bias = None

    def forward(self, in_dict, rbf=None, rij=None):
        out_dict = {}
        for rank in self.ranks:  # [batch, pts, n_chan, rep]
            out_dict[rank] = torch.einsum('ijkl,mk->ijml', in_dict[rank], self.weights[rank])
        if self.bias is not None:
            out_dict[0] += self.bias
        return out_dict, rbf, rij


class RotEquiv_Nonlin(nn.Module):
    def __init__(self, channels, nonlin=ssp, b_init=None):
        super(RotEquiv_Nonlin, self).__init__()
        if b_init is None:
            b_init = init.zeros_

        self.channels = channels
        self.nonlin = nonlin

        self.bias = nn.Parameter(torch.Tensor(channels))
        self.register_parameter('bias', self.bias)
        b_init(self.bias)

    def forward(self, tensor):
        # tensor is [points, channels, rep_idxs]
        if tensor.size(-1) == 1:  # Rank 0 tensors
            return self.nonlin(tensor)

        else:
            norm = eps_l2_norm(tensor, dim=-1)
            nonlin_out = self.nonlin(norm + self.bias)
            factor = nonlin_out / norm
            return tensor * torch.unsqueeze(factor, -1)


class Nonlinearity(nn.Module):
    def __init__(self, channels, nonlin=ssp, b_init=None):
        super(Nonlinearity, self).__init__()
        # channels is a dict: { rank : channels }
        self.ranks = list(channels.keys())
        self.nonlins = { k : RotEquiv_Nonlin(v, nonlin=nonlin, b_init=b_init) for k, v in channels.items() }

    def forward(self, in_dict, rbf=None, rij=None):
        out_dict = {}
        for rank in self.ranks:
            out_dict[rank] = self.nonlins[rank](in_dict[rank])
        return out_dict, rbf, rij


class TFN(nn.Sequential):
    def __init__(self, *args):
        super(TFN, self).__init__(*args)
    
    def forward(self, *args):
        for module in self.children():
            args = module(*args)
        return args

