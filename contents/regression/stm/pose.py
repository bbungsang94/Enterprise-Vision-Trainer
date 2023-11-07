import numpy as np
import torch


def get_t(batch_size):
    p = np.zeros((batch_size, 55, 3))

    return torch.from_numpy(p.reshape((batch_size, -1)))


def get_hands_on(batch_size):
    p = np.zeros((batch_size, 55, 3))

    p[:, 13, 2] = 0.5
    p[:, 14, 2] = -0.5
    p[:, 16, 2] = 1.0
    p[:, 17, 2] = -1.0
    #
    p[:, 18, 0] = -0.5
    p[:, 19, 0] = -0.5
    p[:, 20, 0] = -1.0
    p[:, 21, 0] = -1.0

    return torch.from_numpy(p.reshape((batch_size, -1)))


def get_standing(batch_size):
    p = np.zeros((batch_size, 55, 3))

    p[:, 13, 0] = -0.2
    p[:, 14, 0] = -0.2
    p[:, 13, 2] = -0.4
    p[:, 14, 2] = 0.4
    p[:, 16, 2] = -1.0
    p[:, 17, 2] = 1.0
    return torch.from_numpy(p.reshape((batch_size, -1)))


def get_curve(batch_size):
    p = np.zeros((batch_size, 55, 3))
    # p[:, 1, 0] = -1.5
    # p[:, 2, 0] = -1.5
    # p[:, 4, 0] = 1.5
    # p[:, 5, 0] = 1.5

    p[:, 13, 2] = -0.5
    p[:, 14, 2] = 0.5
    p[:, 16, 2] = -0.75
    p[:, 17, 2] = 0.75

    p[:, 18, 1] = -1.5
    p[:, 19, 1] = 1.5
    return torch.from_numpy(p.reshape((batch_size, -1)))


def get_sitdown(batch_size):
    p = np.zeros((batch_size, 55, 3))
    p[:, 1, 0] = -1.5
    p[:, 2, 0] = -1.5
    p[:, 4, 0] = 1.5
    p[:, 5, 0] = 1.5

    p[:, 13, 2] = -0.5
    p[:, 14, 2] = 0.5
    p[:, 16, 2] = -0.75
    p[:, 17, 2] = 0.75

    p[:, 18, 1] = -1.5
    p[:, 19, 1] = 1.5
    return torch.from_numpy(p.reshape((batch_size, -1)))
