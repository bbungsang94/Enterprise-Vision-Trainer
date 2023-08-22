import copy
import json
import time

import torchvision
from torch import linalg
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import T_co
from torch_geometric.data import Data, Batch
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np
import torch

from contents.reconstruction.pinning.pins.pin import PinLoader


class FLAEPDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn

    @staticmethod
    def _collate_fn(batch) -> [torch.tensor, torch.tensor]:
        g_dict = batch[0][1]
        batch_graph = dict()
        batch_images = []
        batch_shape = []
        batch_expression = []
        batch_gender = []
        batch_jaw = []

        for stub in batch:
            batch_images.append(stub[0])
            batch_gender.append(stub[2])
            batch_shape.append(stub[3])
            batch_expression.append(stub[4])
            batch_jaw.append(stub[5])

        for key in g_dict.keys():
            data_list = []
            for stub in batch:
                data_list.append(stub[1][key])
            g_batch = Batch.from_data_list(data_list=copy.deepcopy(data_list))
            batch_graph[key] = g_batch

        batch_shape = torch.stack(batch_shape, dim=0)
        batch_expression = torch.stack(batch_expression, dim=0)
        batch_jaw = torch.stack(batch_jaw, dim=0)
        return (torch.stack(batch_images, dim=0), batch_graph, batch_gender), [batch_shape, batch_expression, batch_jaw]


class FLAEPNoPinLoader(FLAEPDataLoader):
    @staticmethod
    def _collate_fn(batch) -> [torch.tensor, torch.tensor]:
        g_dict = batch[0][1]
        batch_graph = dict()
        batch_images = []
        batch_gender = []

        # labels
        batch_shape = []
        batch_expression = []
        batch_jaw = []

        for stub in batch:
            batch_images.append(stub[0])
            batch_gender.append(stub[2])
            batch_shape.append(stub[-1]['shape'])
            batch_expression.append(stub[-1]['expression'])
            batch_jaw.append(stub[-1]['jaw'])

        for key in g_dict.keys():
            data_list = []
            for stub in batch:
                data_list.append(stub[1][key])
            g_batch = Batch.from_data_list(data_list=copy.deepcopy(data_list))
            batch_graph[key] = g_batch

        batch_images = torch.stack(batch_images, dim=0)
        batch_shape = torch.stack(batch_shape, dim=0)
        batch_expression = torch.stack(batch_expression, dim=0)
        batch_jaw = torch.stack(batch_jaw, dim=0)
        return (batch_images, batch_graph, batch_gender), [batch_shape, batch_expression, batch_jaw]


class FLAEPDataset(Dataset):
    def __init__(self, dataset_root, labels, edge_path, pin_info):
        self.root = dataset_root
        if '.txt' in labels:
            with open(os.path.join(dataset_root, 'label.txt'), "r") as f:
                labels = f.readlines()
            labels = [label.replace('\n', '') for label in labels]

        self.x_data = labels
        with open(os.path.join(edge_path, "edge_info.json"), "r") as f:
            edge_info = json.load(f)
        edges = dict()
        for key, value in edge_info.items():
            edge = pd.read_csv(os.path.join(edge_path, value), header=None)
            edges[key] = edge.to_numpy()

        self.edges = edges
        self.pin_boxes = PinLoader.load_pins(path=pin_info, filename='pin_info.json')

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        line = self.x_data[idx]
        stub = line.split(' ')
        image = read_image(os.path.join(self.root, stub[0])).type(torch.FloatTensor)
        mean, std = image.mean(), image.std()
        gender = stub[1]
        normalize = torchvision.transforms.Normalize(mean, std)
        image = normalize(image)

        points = pd.read_csv(os.path.join(self.root, stub[-1]), header=None)
        points = points.to_numpy()
        graphs = dict()
        for key, value in self.edges.items():
            origin = None
            for i in range(2, value.shape[1] - 1, 2):
                dump = value[:, i - 2:value.shape[1] - 2]
                nodes = points[dump[:, 0]]
                # check non-cyclic
                if dump[0, 0] != dump[-1, 1]:
                    nodes = np.concatenate((nodes, np.expand_dims(points[dump[-1, 1]], axis=0)), axis=0)
                if origin is not None:
                    origin = np.concatenate((origin, nodes), axis=1)
                else:
                    origin = nodes
            data = Data(x=torch.tensor(origin, dtype=torch.float),
                        edge_index=torch.tensor(value[:, -2:], dtype=torch.long).t().contiguous())
            graphs[key] = copy.deepcopy(data)
        shape, expression, jaw = self.calc_distance(torch.FloatTensor(points))
        return image, graphs, gender, shape, expression, jaw

    def calc_distance(self, landmark, mode='68points'):
        for key, box in self.pin_boxes.items():
            box.switch_mode(mode)
            self.pin_boxes[key] = box

        eyes_box = self.pin_boxes['Eyes']
        nose_box = self.pin_boxes['Nose']
        lip_box = self.pin_boxes['Lip']
        eyebrow_box = self.pin_boxes['Eyebrow']

        # 코 중심, 턱 끝
        nose_tip = landmark[nose_box()[0]]
        jaw_tip = landmark[self.pin_boxes['Align']()[1]]
        # 눈, 입 정보
        eyes_points = landmark[eyes_box()]
        lip_points = landmark[lip_box()]
        eyebrow_points = landmark[eyebrow_box()]

        # 코 중심으로부터의 턱 아래의 거리 - 입이 벌려진 높이
        lip_height = linalg.norm(lip_points[2] - lip_points[3])
        base = linalg.norm(jaw_tip - nose_tip) - lip_height

        # 코 중심으로부터의 눈가 거리
        shape = linalg.norm(eyes_points[0] - nose_tip) / base
        e2 = linalg.norm(eyes_points[1] - nose_tip) / base
        shape = torch.cat((shape.unsqueeze(0), e2.unsqueeze(0)))
        e3 = linalg.norm(eyes_points[4] - nose_tip) / base
        shape = torch.cat((shape, e3.unsqueeze(0)))
        e4 = linalg.norm(eyes_points[5] - nose_tip) / base
        shape = torch.cat((shape, e4.unsqueeze(0)))

        # 코 중심으로부터의 입 양 끝 거리
        l1 = linalg.norm(lip_points[0] - nose_tip) / base
        shape = torch.cat((shape, l1.unsqueeze(0)))
        expression = l1
        jaw = l1
        l2 = linalg.norm(lip_points[1] - nose_tip) / base
        shape = torch.cat((shape, l2.unsqueeze(0)))
        expression = torch.cat((expression.unsqueeze(0), l2.unsqueeze(0)))
        jaw = torch.cat((jaw.unsqueeze(0), l2.unsqueeze(0)))

        # 입의 벌린 정도, 입의 길이
        jaw = torch.cat((jaw, (lip_height / base).unsqueeze(0)))
        l3 = linalg.norm(lip_points[0] - lip_points[1]) / base
        shape = torch.cat((shape, l3.unsqueeze(0)))
        expression = torch.cat((expression, l3.unsqueeze(0)))

        # 눈의 높이(2, 3), 길이(0, 1)
        e5 = linalg.norm(eyes_points[1] - eyes_points[0]) / base
        expression = torch.cat((expression, e5.unsqueeze(0)))
        e6 = linalg.norm(eyes_points[2] - eyes_points[3]) / base
        expression = torch.cat((expression, e6.unsqueeze(0)))
        e7 = linalg.norm(eyes_points[5] - eyes_points[4]) / base
        expression = torch.cat((expression, e7.unsqueeze(0)))
        e8 = linalg.norm(eyes_points[6] - eyes_points[7]) / base
        expression = torch.cat((expression, e8.unsqueeze(0)))

        # 코 중심으로부터의 눈썹의 거리
        b1 = linalg.norm(eyebrow_points[1] - nose_tip) / base
        expression = torch.cat((expression, b1.unsqueeze(0)))
        b2 = linalg.norm(eyebrow_points[3] - nose_tip) / base
        expression = torch.cat((expression, b2.unsqueeze(0)))
        b3 = linalg.norm(eyebrow_points[5] - nose_tip) / base
        expression = torch.cat((expression, b3.unsqueeze(0)))
        b4 = linalg.norm(eyebrow_points[7] - nose_tip) / base
        expression = torch.cat((expression, b4.unsqueeze(0)))

        return shape, expression, jaw


class FLAEPNoPinDataset(FLAEPDataset):
    def __getitem__(self, idx):
        line = self.x_data[idx]
        stub = line.split(' ')
        index = stub[0]
        image = read_image(os.path.join(self.root, 'image', index + '.jpg')).type(torch.FloatTensor)
        mean, std = image.mean(), image.std()
        gender = stub[1]
        normalize = torchvision.transforms.Normalize(mean, std)
        image = normalize(image)

        with open(os.path.join(self.root, 'parameters', index + '.json'), 'r') as f:
            params = json.load(f)

        for key, value in params.items():
            params[key] = torch.tensor(value, dtype=torch.float)
        graph_points = pd.read_csv(os.path.join(self.root, 'landmark-468', index + '.csv'), header=None)
        graph_points = graph_points.to_numpy()
        graphs = dict()
        for key, value in self.edges.items():
            origin = None
            for i in range(2, value.shape[1] - 1, 2):
                dump = value[:, i - 2:value.shape[1] - 2]
                nodes = graph_points[dump[:, 0]]
                # check non-cyclic
                if dump[0, 0] != dump[-1, 1]:
                    nodes = np.concatenate((nodes, np.expand_dims(graph_points[dump[-1, 1]], axis=0)), axis=0)
                if origin is not None:
                    origin = np.concatenate((origin, nodes), axis=1)
                else:
                    origin = nodes
            data = Data(x=torch.tensor(origin, dtype=torch.float),
                        edge_index=torch.tensor(value[:, -2:], dtype=torch.long).t().contiguous())
            graphs[key] = copy.deepcopy(data)
        return image, graphs, gender, params
