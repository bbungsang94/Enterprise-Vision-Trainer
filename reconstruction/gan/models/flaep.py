from typing import Dict, List

import torch
import timm
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential


class FLAEP(nn.Module):
    def __init__(self):
        super().__init__()
        # output: batch, 1280
        self.latent_model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        o = self.latent_model(torch.randn(1, 3, 1024, 1024))
        residual_module = nn.Sequential(nn.Linear(o.shape[-1], 1024),
                                        nn.PReLU(),
                                        nn.Linear(1024, 512),
                                        nn.PReLU(),
                                        nn.Linear(512, 256))

        outline_gcn = GCNFlaep(in_channel=3, out_channel=16)
        eyes_gcn = GCNFlaep(in_channel=6, out_channel=32)
        lips_gcn = GCNFlaep(in_channel=6, out_channel=16)
        borrow_gcn = GCNFlaep(in_channel=6, out_channel=32)

        shape_head = nn.Sequential(nn.Linear(1664, 1664 // 4),
                                   nn.Tanh(),
                                   nn.Linear(1664 // 4, 1664 // 8),
                                   nn.Tanh(),
                                   nn.Linear(1664 // 8, 20),
                                   nn.Tanh())

        expression_head = nn.Sequential(nn.Linear(864, 864 // 4),
                                        nn.Tanh(),
                                        nn.Linear(864 // 4, 864 // 8),
                                        nn.Tanh(),
                                        nn.Linear(864 // 8, 10),
                                        nn.Tanh())

        jaw_head = nn.Sequential(nn.Linear(1152, 1152 // 8),
                                 nn.Tanh(),
                                 nn.Linear(1152 // 8, 1152 // 16),
                                 nn.Tanh(),
                                 nn.Linear(1152 // 16, 3),
                                 nn.Tanh())

        neck_head = nn.Sequential(nn.Linear(1152, 1152 // 8),
                                  nn.Tanh(),
                                  nn.Linear(1152 // 8, 1152 // 16),
                                  nn.Tanh(),
                                  nn.Linear(1152 // 16, 3),
                                  nn.Tanh())

        self.Bodies = nn.ModuleDict(
            {
                'Residual': residual_module,
                'Outline': outline_gcn,
                'Eyes': eyes_gcn,
                'Borrow': borrow_gcn,
                'Lips': lips_gcn,
                'ShapeHead': shape_head,
                'ExpHead': expression_head,
                'JawHead': jaw_head,
                'NeckHead': neck_head
            }
        )

    def forward(self, image, graphs):
        latent_space = self.latent_model(image)
        res_latent = self.Bodies['Residual'](latent_space)

        num_graph = graphs['Outline'].num_graphs
        outline = self.Bodies['Outline'](graphs['Outline'])
        outline = outline.view(num_graph, -1)
        eyes = self.Bodies['Eyes'](graphs['Eyes'])
        eyes = eyes.view(num_graph, -1)
        borrow = self.Bodies['Borrow'](graphs['Borrow'])
        borrow = borrow.view(num_graph, -1)
        lips = self.Bodies['Lips'](graphs['Lips'])
        lips = lips.view(num_graph, -1)

        shape = torch.cat([res_latent, lips, eyes, outline], dim=1)
        expression = torch.cat([res_latent, lips, borrow], dim=1)
        rot = torch.cat([res_latent, lips, outline], dim=1)

        shape = self.Bodies['ShapeHead'](shape)
        expression = self.Bodies['ExpHead'](expression)
        jaw = self.Bodies['JawHead'](rot)
        jaw = torch.cat([torch.zeros(num_graph, 3, device=jaw.device), jaw], dim=1)
        neck = self.Bodies['NeckHead'](rot)

        return {'shape_params': shape, 'expression_params': expression, 'pose_params': jaw, 'neck_pose': neck}


class GCNFlaep(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = Sequential('x, edge_index',
                                [(GCNConv(in_channels=in_channel, out_channels=8),
                                  'x=x, edge_index=edge_index -> x'),
                                 nn.LeakyReLU(),
                                 (GCNConv(in_channels=8, out_channels=16),
                                  'x=x, edge_index=edge_index -> x'),
                                 nn.LeakyReLU(),
                                 nn.Dropout(),
                                 (GCNConv(in_channels=16, out_channels=out_channel),
                                  'x=x, edge_index=edge_index -> x'),
                                 nn.LeakyReLU(),
                                 nn.Dropout(),
                                 ])

    def forward(self, x):
        return self.model(x.x, x.edge_index)


if __name__ == "__main__":
    test = FLAEP()
