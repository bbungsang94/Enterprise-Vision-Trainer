from typing import Tuple

import torch
from torch import nn

from modules.blocks import EncoderBlock, DecoderBlock, DoubleConv


class ClassicUnet(nn.Module):
    def __init__(self, colour, n_of_blocks: Tuple[int, int] = (4, 4), **kwargs):
        super().__init__()
        pivot = 6
        encoder = [EncoderBlock(in_channels=colour, out_channels=2 ** pivot)]
        for t in range(n_of_blocks[0] - 1):
            encoder.append(EncoderBlock(in_channels=2 ** (pivot + t), out_channels=2 ** (pivot + t + 1)))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for t in reversed(range(n_of_blocks[1])):
            decoder.append(DecoderBlock(in_channels=2 ** (pivot + t + 1), out_channels=2 ** (pivot + t)))
        self.decoder = nn.ModuleList(decoder)

        self.bridge = DoubleConv(in_channels=2 ** (pivot + n_of_blocks[0] - 1),
                                 out_channels=2 ** (pivot + n_of_blocks[1]))

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=2 ** pivot, out_channels=colour, kernel_size=(3, 3), padding="same"),
            nn.Sigmoid()
        )
        self.iteration = n_of_blocks

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        s1, p1 = self.encoder[0](x)
        s2, p2 = self.encoder[1](p1)
        s3, p3 = self.encoder[2](p2)
        s4, p4 = self.encoder[3](p3)

        bridge = self.bridge(p4)

        d = self.decoder[0](bridge, s4)
        d = self.decoder[1](d, s3)
        d = self.decoder[2](d, s2)
        d = self.decoder[3](d, s1)

        output = self.head(d)
        return bridge[:, :3], output
