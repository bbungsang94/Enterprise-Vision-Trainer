import torch
from torch import nn
from modules.embedding import sinusoidal_embedding
from modules.layer import LayerNormBlock, simple_mlp


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
            LayerNormBlock((3, 256, 256), 3, 10),
            LayerNormBlock((10, 256, 256), 10, 10),
            LayerNormBlock((10, 256, 256), 10, 10)
        )
        self.b2 = nn.Sequential(
            LayerNormBlock((10, 128, 128), 10, 20),
            LayerNormBlock((20, 128, 128), 20, 20),
            LayerNormBlock((20, 128, 128), 20, 20)
        )
        self.b3 = nn.Sequential(
            LayerNormBlock((20, 64, 64), 20, 40),
            LayerNormBlock((40, 64, 64), 40, 40),
            LayerNormBlock((40, 64, 64), 40, 40)
        )
        # endregion

        # region Bottleneck
        self.te_mid = simple_mlp(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            LayerNormBlock((40, 32, 32), 40, 20),
            LayerNormBlock((20, 32, 32), 20, 20),
            LayerNormBlock((20, 32, 32), 20, 40)
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
            LayerNormBlock((80, 64, 64), 80, 40),
            LayerNormBlock((40, 64, 64), 40, 20),
            LayerNormBlock((20, 64, 64), 20, 20)
        )
        self.b5 = nn.Sequential(
            LayerNormBlock((40, 128, 128), 40, 20),
            LayerNormBlock((20, 128, 128), 20, 10),
            LayerNormBlock((10, 128, 128), 10, 10)
        )
        self.b_out = nn.Sequential(
            LayerNormBlock((20, 256, 256), 20, 10),
            LayerNormBlock((10, 256, 256), 10, 10),
            LayerNormBlock((10, 256, 256), 10, 10, normalize=False)
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

    # def summary(self):
    #     model, images,
    #     betas, alphas, alpha_bars,
    #     frames_per_gif, gif_name,
    #     save_path