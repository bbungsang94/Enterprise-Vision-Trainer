import math
from typing import Tuple

import torch
from torch import nn
from torchvision.io import write_jpeg
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from modules.blocks import DoubleConv, TransposedDoubleConv


class CelebAVAE(nn.Module):
    def __init__(self):
        super(CelebAVAE, self).__init__()

        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, 218, 178))
        self.size = out.flatten().shape[0]
        self.fc_mu = nn.Linear(self.size, 128)
        self.fc_var = nn.Linear(self.size, 128)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(128, self.size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

        self.trans = transforms.Compose([
            transforms.Resize((218, 178), antialias=True),
            transforms.CenterCrop((218, 178))])  # used by decode method to transform final output

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, 7, 6)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.trans(result)
        result = torch.nan_to_num(result)
        return result

    def forward(self, x) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        result = self.encoder(x)
        flatten = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(flatten)
        log_var = self.fc_var(flatten)
        log_var = torch.clamp_(log_var, -10, 10)

        z = self.reparameterize(mu=mu, logvar=log_var)
        return result[:, :3], (self.decode(z), mu, log_var)


class GenderAgeAE(nn.Module):
    def __init__(self):
        super(GenderAgeAE, self).__init__()

        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, 218, 178))
        self.size = out.flatten().shape[0]
        self.fc = nn.Linear(self.size, 128)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(128, self.size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def encode(self, x):
        result = self.encoder(x)
        flatten = torch.flatten(result, start_dim=1)
        z = self.fc(flatten)
        return z

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z[:, :3], self.decoder(z)


def test():
    import os
    root = r"D:\Creadto\Utilities\Enterprise-Vision-Trainer\output\checkpoint\VanillaVAE\00000077\00001999"
    model_name = r"20240103182604.pth"
    load_state = torch.load(os.path.join(root, model_name))
    model = CelebAVAE()
    model.load_state_dict(load_state['model'])
    model.eval()
    with torch.no_grad():
        latent = torch.randn(16, 128)
        result = model.decode(latent)

    grid = make_grid(result, nrow=int(math.sqrt(len(result)))).cpu() * 255
    write_jpeg(grid.type(dtype=torch.uint8), 'test.jpeg')
    #


if __name__ == "__main__":
    test()
