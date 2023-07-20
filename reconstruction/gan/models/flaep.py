from typing import Dict, List

import torch
import timm
import torch.nn as nn
from torch_geometric.nn import GCNConv


class FLAEP(nn.Module):
    def __init__(self, edges: Dict[torch.tensor]):
        super().__init__()
        # output: batch, 1280
        self.latent_model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        o = self.latent_model(torch.randn(16, 3, 1024, 1024))
        self.edge_index = edges
        residual_module = nn.Sequential(nn.Linear(o.shape[-1], 1024),
                                        nn.PReLU(),
                                        nn.Linear(1024, 512),
                                        nn.PReLU(),
                                        nn.Linear(512, 256))

        shape_gcn = GCNFlaep(in_channel=4, out_channel=64)
        expression_gcn = GCNFlaep(in_channel=2, out_channel=64)
        rot_gcn = GCNFlaep(in_channel=2, out_channel=32)

        shape_head = nn.Sequential(nn.Linear(512, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, 20),
                                   nn.Tanh())

        expression_head = nn.Sequential(nn.Linear(512, 128),
                                        nn.Tanh(),
                                        nn.Linear(128, 20),
                                        nn.Tanh())

        jaw_head = nn.Sequential(nn.Linear(512, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 4),
                                 nn.Tanh())

        neck_head = nn.Sequential(nn.Linear(512, 32),
                                  nn.Tanh(),
                                  nn.Linear(32, 3),
                                  nn.Tanh())

        self.Bodies = nn.ModuleDict(
            {
                'Residual': residual_module,
                'ShapeGCN': shape_gcn,
                'ExpGCN': expression_gcn,
                'RotGCN': rot_gcn,
                'ShapeHead': shape_head,
                'ExpHead': expression_head,
                'JawHead': jaw_head,
                'NeckHead': neck_head
            }
        )

    def forward(self, image, **graphs):
        latent_space = self.latent_model(image)
        res_latent = self.Bodies['Residual'](latent_space)

        shape = self.Bodies['ShapeGCN'](graphs['shape'])
        expression = self.Bodies['ExpGCN'](graphs['expression'])
        rot = self.Bodies['RotGCN'](graphs['rotation'])

        shape = torch.cat([res_latent, shape])
        expression = torch.cat([res_latent, expression])
        rot = torch.cat([res_latent, rot])

        shape = self.Bodies['ShapeHead'](shape)
        expression = self.Bodies['ExpHead'](expression)
        jaw = self.Bodies['JawHead'](rot)
        neck = self.Bodies['NeckHead'](rot)

        return {'shape': shape, 'expression': expression, 'jaw': jaw, 'neck': neck}


class GCNFlaep(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(
            GCNConv(in_channels=in_channel, out_channels=16),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            GCNConv(in_channels=16, out_channels=32),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(),
            GCNConv(in_channels=32, out_channels=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(4),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    test = FLAEP()
