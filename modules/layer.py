from typing import Dict, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from modules.blocks import DoubleConv
from modules.utils import default


# small helper modules

def upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


def simple_mlp(dim_in, dim_out, activation="SiLU"):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        getattr(nn, activation)(),
        nn.Linear(dim_out, dim_out)
    )


class DoubleConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x


class DoubleConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(x)


class IdentityLayer(nn.Module):
    def __init__(self, model_config: Dict[str, Any], network_metadata: Dict[str, Any]):
        super().__init__()
        self.model_config = model_config
        self.network_metadata = network_metadata

    def forward(self, decoder_output):
        x = decoder_output[0]
        return x


class PixelShuffleUpsample(IdentityLayer):
    def __init__(self, model_config: Dict[str, Any], network_metadata: Dict[str, Any]):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        is_coreml = model_config.get("is_coreml", False)
        self.shuffle = PixelShuffle(upscale_factor=4)

    def forward(self, decoder_output):
        x = decoder_output[0]
        x = self.shuffle(x)
        return x
