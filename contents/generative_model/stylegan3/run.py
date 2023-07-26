import pickle
from typing import Tuple

import torch
import matplotlib.pyplot as pyplot
import numpy as np


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def main(translate, rotate, load_path=r"./pretrained/stylegan3-r-ffhq-1024x1024.pkl"):
    device = torch.device('cuda')

    with open(load_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
    z = torch.randn([1, G.z_dim]).to(device)  # latent codes
    label = None

    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    # img = G(z, label, truncation_psi=0.5, noise_mode='const')

    w = G.mapping(z, label, truncation_psi=0.5, truncation_cutoff=8)
    img = G.synthesis(w, noise_mode='const', force_fp32=True)
    img = img.clamp_(-1, 1).add_(1).div_(2.0)
    img = img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()

    # input_z = z.detach().squeeze().cpu().view(2, -1), permute(1, 2, 0).numpy()
    pyplot.imshow(img)


if __name__ == "__main__":
    main(translate=(0.0, 0.0), rotate=0.0)
