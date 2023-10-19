import torch
import torch.nn as nn

from modules.blocks import DoubleConv, EncoderBlock


class BasicLandmarker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.SiLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Dropout(),
        )

        self.head_x = nn.Sequential(
            nn.Linear(1024, 478),
            nn.Sigmoid()
        )
        self.head_y = nn.Sequential(
            nn.Linear(1024, 478),
            nn.Sigmoid()
        )
        self.head_z = nn.Sequential(
            nn.Linear(1024, 478),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = latent.view(len(x), -1)
        x = self.decoder(x)
        o = torch.stack([self.head_x(x), self.head_y(x), self.head_z(x)], dim=2)
        return latent[:, :3], o
