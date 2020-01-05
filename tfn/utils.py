import numpy as np
import scipy.linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
LOG_2 = 0.69314718056
EPS_TENSOR = torch.tensor(EPS)

# 3D Levi-Civita tensor
EIJK = torch.zeros(3, 3, 3, requires_grad=False)
EIJK[0, 1, 2] = EIJK[1, 2, 0] = EIJK[2, 0, 1] = 1
EIJK[0, 2, 1] = EIJK[2, 1, 0] = EIJK[1, 0, 2] = -1


def eps_l2_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(torch.max(torch.sum(tensor**2), EPS_TENSOR))

    return torch.sqrt(torch.max(torch.sum(tensor**2, dim=dim, keepdim=keepdim), EPS_TENSOR))


def ssp(x):
    return F.softplus(x) - LOG_2


def diff_matrix(geom):
    ri = torch.unsqueeze(geom, -2)  # [batch, N, 1, 3]
    rj = torch.unsqueeze(geom, -3)  # [batch, 1, N, 3]
    return ri - rj  # Broadcasted to [batch, N, N, 3]


def dist_matrix(geom):
    return eps_l2_norm(diff_matrix(geom), dim=-1)  # [batch, N, N]


def rot_matrix(axis, theta):
    return LA.expm(np.cross(np.eye(3), axis * theta))


def rand_rot_batch(n=1, np_random=None):
    if np_random is None:
        np_random = np.random

    axis = np_random.randn(n, 3)
    norm = np.expand_dims(LA.norm(axis, ord=2, axis=1) + EPS, -1)
    axis = axis / norm
    theta = np.expand_dims(2 * np.pi * np_random.uniform(0.0, 1.0, size=n), -1)
    output = np.zeros((n, 3, 3))
    for i, (a, t) in enumerate(zip(axis, theta)):
        output[i, :, :] = rot_matrix(a, t)
    return torch.tensor(output).float()

