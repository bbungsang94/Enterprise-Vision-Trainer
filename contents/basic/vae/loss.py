import torch
from torch.nn import functional as F


class VAELoss:
    def __init__(self):
        pass

    def __call__(self, recon_pack, x):
        recons, mu, logvar = recon_pack
        recon_loss = F.mse_loss(recons, x)
        if recons.requires_grad:
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kld_weight = 0.00025
            kld_loss = kld_weight * KLD
        else:
            kld_loss = 0.

        return recon_loss + kld_loss
