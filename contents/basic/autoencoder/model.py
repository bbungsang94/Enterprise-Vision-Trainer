import torch
import torch.nn as nn

from modules.blocks import DoubleConv
from modules.layer import DoubleConvDown, DoubleConvUp


class StaticAutoencdoer(nn.Module):
    def __init__(self, colour, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(colour, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, colour, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> [torch.Tensor]:
        z = self.encoder(x)
        o = self.decoder(z)
        return z[:, :3], o


class Autoencdoer4x(nn.Module):
    def __init__(self, colour, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(colour, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, colour, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> [torch.Tensor]:
        # 872 712
        z = self.encoder(x)
        o = self.decoder(z)
        return o


class DoubleEncoder(nn.Module):
    def __init__(self, colour, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(colour, 64, 12, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 8, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=8, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, colour, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> [torch.Tensor]:
        z = self.encoder(x)
        o = self.decoder(z)
        return z[:, :3], o
