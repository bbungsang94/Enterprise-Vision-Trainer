import copy
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import T_co
from torch_geometric.data import Data, Batch
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np
import torch


class FLAEPDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn

    @staticmethod
    def _collate_fn(batch):
        g_dict = batch[0][1]
        batch_graph = dict()
        batch_landmarks = []
        batch_images = []

        for stub in batch:
            batch_images.append(stub[0])
            batch_landmarks.append(stub[2])

        for key in g_dict.keys():
            data_list = []
            for stub in batch:
                data_list.append(stub[1][key])
            g_batch = Batch.from_data_list(data_list=copy.deepcopy(data_list))
            batch_graph[key] = g_batch

        return torch.stack(batch_images, dim=0), batch_graph, torch.stack(batch_landmarks, dim=0)


class FLAEPDataset(Dataset):
    def __init__(self, dataset_root, labels, edge_info):
        self.root = dataset_root
        self.x_data = labels
        self.edges = edge_info
        self._target = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self._target)

    def set_device(self, device: str):
        self._target = device
        self.device = torch.device(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        line = self.x_data[idx]
        stub = line.split(' ')
        image = read_image(os.path.join(self.root, stub[0])).type(torch.FloatTensor)
        mean, std = image.mean(), image.std()

        normalize = torchvision.transforms.Normalize(mean, std)
        image = normalize(image)

        points = pd.read_csv(os.path.join(self.root, stub[-1]))
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
            graphs[key] = copy.deepcopy(data).to(self._target)
        return image.to(self.device), graphs, torch.FloatTensor(points).to(self.device)
