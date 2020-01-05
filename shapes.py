import numpy as np
import random
from math import pi, sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from tfn.layers import Filter, Convolution, SelfInteraction, Nonlinearity, TFN
from tfn.utils import ssp, diff_matrix, dist_matrix, rand_rot_batch

# Parameters
# RBF
rbf_low = 0.0
rbf_high = 3.5
rbf_count = 4
# Training
n_epochs = 2000
print_freq = 100
lr = 1e-3
# Testing
test_epochs = 10

# 3D Tetris dataset
tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [np.array(points_) for points_ in tetris]
num_classes = len(dataset)

# Define network modules:
modules = []
# Self-Interaction 0
modules.append(SelfInteraction({0 : 1}, {0 : 1}, bias=False))

# Convolution 1
filters = {
    0 : {
        0 : { 0 : Filter(rbf_count, 1, 0, 0, 0) },
        1 : { 1 : Filter(rbf_count, 1, 0, 1, 1) },
    },
}
modules.append(Convolution(rbf_count, 1, [0], [0, 1], filters=filters))
# Self-Interaction 1
modules.append(SelfInteraction({0 : 1, 1 : 1}, {0 : 4, 1 : 4}))
# Nonlinearity 1
modules.append(Nonlinearity({0 : 4, 1 : 4}))

# Convolution 2
filters = {
    0 : {
        0 : { 0 : Filter(rbf_count, 4, 0, 0, 0) },
        1 : { 1 : Filter(rbf_count, 4, 0, 1, 1) },
    },
    1 : {
        0 : { 1 : Filter(rbf_count, 4, 1, 0, 1) },
        1 : { 0 : Filter(rbf_count, 4, 1, 1, 0),
              1 : Filter(rbf_count, 4, 1, 1, 1) },
    }
}
modules.append(Convolution(rbf_count, 4, [0, 1], [0, 1], filters=filters))
# Self-Interaction 2
modules.append(SelfInteraction({0 : 8, 1 : 12}, {0 : 4, 1 : 4}))
# Nonlinearity 2
modules.append(Nonlinearity({0 : 4, 1 : 4}))

# Convolution 3
filters = {
    0 : {
        0 : { 0 : Filter(rbf_count, 4, 0, 0, 0) }
    },
    1 : {
        0 : { 1 : Filter(rbf_count, 4, 1, 0, 1) }
    }
}
modules.append(Convolution(rbf_count, 4, [0, 1], [0], filters=filters))
# Self-Interaction 3
modules.append(SelfInteraction({0 : 8}, {0 : 4}))
# Nonlinearity 3
modules.append(Nonlinearity({0 : 4}))

# NETWORK
tfn = TFN(*modules)
fc = nn.Linear(4, num_classes)
init.kaiming_normal_(fc.weight)
init.zeros_(fc.bias)

# RBF
rbf_spacing = (rbf_high - rbf_low) / rbf_count
rbf_centers = torch.linspace(rbf_low, rbf_high, rbf_count)
gamma = 1 / rbf_spacing

points = torch.Tensor(dataset)
rij = diff_matrix(points)
dij = dist_matrix(points)
rbf = torch.exp(-gamma * (torch.unsqueeze(dij, -1) - rbf_centers)**2)

# Input tensor
in_dict = { 0 : torch.ones(points.size(0), points.size(1), 1, 1) }
# Lables
labels = torch.arange(0, 8).long()

# Optimizer
params = list(tfn.parameters()) + list(fc.parameters())
opt = optim.Adam(params, lr=lr)

# TRAINING
for epoch in range(n_epochs + 1):
    out_dict, _, _ = tfn(in_dict, rbf, rij)
    out = out_dict[0]
    out = torch.sum(out, dim=1).squeeze()
    out = fc(out)
    loss = F.cross_entropy(out, labels)
    
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % print_freq == 0:
        print('epoch %i: %f' % (epoch, loss))

rng = np.random.RandomState()

correct = 0
for epoch in range(test_epochs):
    rotations = rand_rot_batch(n=len(dataset), np_random=rng)  # [batch, 3, 3]
    rotated = torch.einsum('ijk,ilj->ilk', rotations, points)

    rij = diff_matrix(rotated)
    dij = dist_matrix(rotated)
    rbf = torch.exp(-gamma * (torch.unsqueeze(dij, -1) - rbf_centers)**2)

    out_dict, _, _ = tfn(in_dict, rbf, rij)
    out = out_dict[0]
    out = torch.sum(out, dim=1).squeeze()
    out = fc(out)
    idx = torch.argmax(out, dim=1)
    correct += torch.sum((idx == labels).int()).item()

print('Test accuracy: %f' % (correct / (test_epochs * len(dataset))))
