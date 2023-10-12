import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from modules.normalization import GroupNorm
from modules.utils import exists


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = GroupNorm(dim, dim_out, groups=groups)
        self.block2 = GroupNorm(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel=(3, 3)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel=(3, 3)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2,
                                           kernel_size=(2, 2), stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x
