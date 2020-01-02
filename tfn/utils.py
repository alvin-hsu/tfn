import numpy as np
import scipy.linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
LOG_2 = 0.69314718056

# 3D Levi-Civita tensor
EIJK = torch.zeros(3, 3, 3, requires_grad=False)
EIJK[0, 1, 2] = EIJK[1, 2, 0] = EIJK[2, 0, 1] = 1
EIJK[0, 2, 1] = EIJK[2, 1, 0] = EIJK[1, 0, 2] = -1


def eps_l2_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(torch.max(torch.sum(tensor**2), EPS))

    else:
        return torch.sqrt(torch.max(torch.sum(tensor**2, dim=dim, keepdim=keepdim), EPS))


def ssp(x):
    return F.softplus(x) - LOG_2


def diff_matrix(geom):
    ri = torch.unsqueeze(geom, 1)  # [N, 1, 3]
    rj = torch.unsqueeze(geom, 0)  # [1, N, 3]
    return ri - rj  # Broadcasted to [N, N, 3]


def dist_matrix(geom):
    return eps_l2_norm(diff_matrix(geom), dim=-1)  # [N, N]


def rand_rot_matrix(np_random=None)
    if np_random is None:
        np_random = np.random

    axis = np_random.randn(3)
    axis /= np.linalg.norm(axis) + EPS
    theta = 2 * np.pi * np_random.uniform(0.0, 1.0)
    return rot_matrix(axis, theta)


def rot_matrix(axis, theta):
    return LA.expm(np.cross(np.eye(3), axis * theta))


def unit_vecs(v, dim=-1):
    return v / eps_l2_norm(v, dim=dim, keepdim=True)


def Y_2(rij):
    # rij = diff_matrix [N, N, 3]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = torch.max(torch.sum(tensor**2, dim=-1), EPS)
    output = torch.stack([
            x * y / r2,
            y * z / r2,
            (2 * z**2 - x**2 - y**2) / (2 * np.sqrt(3) * r2),
            z * x / r2
            (x**2 - y**2) / (2 * r2)
        ], dim=-1)
    return output


