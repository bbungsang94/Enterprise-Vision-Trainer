from typing import Dict, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from modules.utils import default, exists


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


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNormBlock(nn.Module):
    def __init__(self, shape, in_channel, out_channel,
                 kernel_size=3, stride=1, padding=1, activation="SiLU", normalize=True):
        super(LayerNormBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.activation = getattr(nn, activation)()
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class GroupNormBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = GroupNormBlock(dim, dim_out, groups=groups)
        self.block2 = GroupNormBlock(dim_out, dim_out, groups=groups)
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
