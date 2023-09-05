import math
from typing import T, Optional, Union

import torch
from torch import nn

from modules.attention import SAWrapper
from modules.blocks import DoubleConv
from modules.embedding import sinusoidal_embedding
from modules.layer import simple_mlp, DoubleConvUp, DoubleConvDown
from modules.normalization import LayerNorm


class SkinDiffusion(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, min_beta=10 ** -4, max_beta=0.02, colour=True):
        super(SkinDiffusion, self).__init__()
        self.device = torch.device("cpu")

        self.colour = 3 if colour is True else 1
        self.n_steps = n_steps
        self.betas = torch.linspace(min_beta, max_beta, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

        # networks 3 32 32 -> 64 32 32 3 256 256
        bi_linear = True
        self.inc = DoubleConv(self.colour, 64)
        self.down1 = DoubleConvDown(64, 128)
        self.down2 = DoubleConvDown(128, 256)
        factor = 2 if bi_linear else 1
        self.down3 = DoubleConvDown(256, 512 // factor)
        self.up1 = DoubleConvUp(512, 256 // factor, bi_linear)
        self.up2 = DoubleConvUp(256, 128 // factor, bi_linear)
        self.up3 = DoubleConvUp(128, 64, bi_linear)
        self.out = nn.Conv2d(64, self.colour, kernel_size=1)
        self.sa1 = SAWrapper(256, 64)
        self.sa2 = SAWrapper(256, 32)
        self.sa3 = SAWrapper(128, 64)

    def to(self, *args, **kwargs) -> T:
        super(SkinDiffusion, self).to(*args, **kwargs)
        self.device = args[0]
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        ret = pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)
        return ret

    def forward(self, x0) -> [torch.Tensor]:
        sample, epsilon = self.make_noise(x0)
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        e_hat = self.backward(*sample)
        loss = nn.functional.mse_loss(e_hat, epsilon)
        return loss

    def backward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 128)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 64)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 32)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 64)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 128)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 256)
        e_hat = self.out(x)
        return e_hat

    def regenerate(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
                post_sigma = math.sqrt(self.betas[t]) * z
            else:
                post_sigma = torch.tensor(0.0, dtype=torch.float32)
            time_tensor = (torch.ones(x.shape[0], 1) * t).to(self.device).long()

            e_hat = self.backward(x, time_tensor)
            pre_scale = 1 / math.sqrt(self.alphas[t])
            e_scale = (1 - self.alphas[t]) / math.sqrt(1 - self.alpha_bars[t])
            x = pre_scale * (x - e_scale * e_hat) + post_sigma.to(self.device)
            return x

    def make_noise(self, x0):
        n = x0.shape[0]
        t = torch.randint(0, self.n_steps, (n,)).to(self.device)

        # make noise(diffusion forward)
        epsilon = torch.randn(x0.shape).to(self.device)
        a_bar = self.alpha_bars[t]
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * epsilon

        return (noisy, t.unsqueeze(-1).type(torch.float)), epsilon

    def summary(self, skins, images, **kwargs):
        result = {'model': None,
                  'x0': skins,
                  'inputs': images,
                  'frames_per_gif': 100,
                  'gif_name': "generation.gif",
                  'save_path': None,
                  'device': self.device}
        return result


class SkinUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(SkinUNet, self).__init__()
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # region Down-Sampling
        self.te1 = simple_mlp(time_emb_dim, 3)
        self.te2 = simple_mlp(time_emb_dim, 10)
        self.te3 = simple_mlp(time_emb_dim, 20)

        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 2)
        )

        self.b1 = nn.Sequential(
            LayerNorm((3, 256, 256), 3, 10),
            LayerNorm((10, 256, 256), 10, 10),
            LayerNorm((10, 256, 256), 10, 10)
        )
        self.b2 = nn.Sequential(
            LayerNorm((10, 128, 128), 10, 20),
            LayerNorm((20, 128, 128), 20, 20),
            LayerNorm((20, 128, 128), 20, 20)
        )
        self.b3 = nn.Sequential(
            LayerNorm((20, 64, 64), 20, 40),
            LayerNorm((40, 64, 64), 40, 40),
            LayerNorm((40, 64, 64), 40, 40)
        )
        # endregion

        # region Bottleneck
        self.te_mid = simple_mlp(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            LayerNorm((40, 32, 32), 40, 20),
            LayerNorm((20, 32, 32), 20, 20),
            LayerNorm((20, 32, 32), 20, 40)
        )
        # endregion

        # region Up-Sampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 4, 2, 2)
        )
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)

        self.te4 = simple_mlp(time_emb_dim, 80)
        self.te5 = simple_mlp(time_emb_dim, 40)
        self.te_out = simple_mlp(time_emb_dim, 20)

        self.b4 = nn.Sequential(
            LayerNorm((80, 64, 64), 80, 40),
            LayerNorm((40, 64, 64), 40, 20),
            LayerNorm((20, 64, 64), 20, 20)
        )
        self.b5 = nn.Sequential(
            LayerNorm((40, 128, 128), 40, 20),
            LayerNorm((20, 128, 128), 20, 10),
            LayerNorm((10, 128, 128), 10, 10)
        )
        self.b_out = nn.Sequential(
            LayerNorm((20, 256, 256), 20, 10),
            LayerNorm((10, 256, 256), 10, 10),
            LayerNorm((10, 256, 256), 10, 10, normalize=False)
        )
        self.conv_out = nn.Conv2d(10, 3, 3, 1, 1)

    def forward(self, sample) -> [torch.Tensor]:
        x, t = sample
        # x_T is (N, 6, 256, 256) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 256, 256)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 128, 128)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 64, 64)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 32, 32)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 64, 64)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 64, 64)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 128, 128)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 128, 128)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 256, 256)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 3, 256, 256)

        out = self.conv_out(out)

        return out

    @staticmethod
    def summary(images, skins, **kwargs):
        result = {'inputs': images,
                  'x0': skins,
                  'xT': kwargs['xT'],
                  'betas': kwargs['betas'],
                  'alphas': kwargs['alphas'],
                  'alpha_bars': kwargs['alpha_bars'],
                  'n_steps': kwargs['n_steps'],
                  'frames_per_gif': 100,
                  'gif_name': "generation.gif",
                  'device': kwargs['device'],
                  'model': None,
                  'save_path': None}
        return result
