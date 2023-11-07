import torch
import numpy as np
import pandas as pd
from torch import Tensor


def vertex_to_csv(vertex: np.ndarray, filename: str):
    dataframe = pd.DataFrame(vertex)
    dataframe.to_csv(filename, header=False, index=False)


def rodrigues(r: Tensor):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].

    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
    a = r_hat.permute(0, 2, 1)
    dot = torch.matmul(a, r_hat)
    r = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return r


def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float64).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret
