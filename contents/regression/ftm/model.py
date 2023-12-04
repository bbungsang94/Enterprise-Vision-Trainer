import copy
import json
import os

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv

from contents.regression.ftm.convention import get_interactions
from contents.utility.io_utils import ModelPath
from contents.utility.taylor import GraphTaylor
from external.flame.flame import FLAMESet
from modules.layer import MultiHeadGATLayer


class FTMRegression(nn.Module):
    def __init__(self, node_dim, edge_dim, n_of_node, shape_dim, flame_root,
                 encoder_path, pin_root, circ_root, num_heads=5):
        super().__init__()
        self.encoder = MultiHeadGATLayer(in_dim=node_dim, out_dim=16, edge_dim=edge_dim, num_heads=num_heads)
        # encoder_dict = torch.load(encoder_path)
        # self.encoder.load_state_dict(encoder_dict['encoder'])
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        encoded_len = n_of_node * 16 * num_heads
        self.node_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoded_len + int(encoded_len * 0.3), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.edge_regressor = nn.Sequential(
            nn.Linear((44 * edge_dim), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(512, shape_dim),
        )

        # generic_path = ModelPath(root=flame_root,
        #                          filename="generic_model.pkl",
        #                          config=os.path.join(flame_root, "flame.yaml")
        #                          )
        #
        # with open(os.path.join(pin_root, 'facial.json'), 'r', encoding='UTF-8-sig') as f:
        #     facial = json.load(f)
        # with open(os.path.join(circ_root, 'circumference-facial.json'), 'r', encoding='UTF-8-sig') as f:
        #     circ_dict = json.load(f)

        # self.flame = FLAMESet(generic_path, gender=True)
        # self.taylor = GraphTaylor(tape=get_interactions(), pin=[facial], circ_dict=circ_dict)

    def forward(self, data, gender, gender_tensor):
        edge_attr = data.edge_attr
        z = self.encoder(data)
        z = z.view(len(gender), -1)
        g_repeat = gender_tensor.repeat(1, int(z.shape[1] * 0.3))
        z = torch.cat([g_repeat, z], dim=1)
        o_n = self.node_regressor(z)

        edge_attr = edge_attr.view(len(gender), -1)
        o_e = self.edge_regressor(edge_attr)

        o = self.head(torch.cat([o_n, o_e], dim=1))
        return None, o

    def do_taylor(self, shape, gender):
        batch_size = len(gender)
        poses = self._get_poses(batch_size=batch_size)

        models = dict()
        for key, value in poses.items():
            v, _ = self.flame(genders=gender, batch_size=len(gender), shape=shape, expression=value.to(shape.device))
            models[key] = copy.deepcopy(v)

        self.taylor.update(model_dict=models)
        measure = self.taylor.order(gender=gender, visualize=False)
        return measure

    @staticmethod
    def _get_poses(batch_size):
        result = {
            "standard": torch.zeros(batch_size, 100),
        }
        return result
