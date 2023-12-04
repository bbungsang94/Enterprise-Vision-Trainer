import copy
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm


class FTMDataset(Dataset):
    def __init__(self, root):
        x = []
        y = []
        files = os.listdir(root)
        for file in tqdm(files):
            data = torch.load(os.path.join(root, file))
            gender = data['input']['gender']
            if "male" == gender:
                gender_idx = 0
            else:
                gender_idx = 1
            shape = data['input']['shape']
            graph = data['output']['graph']
            x.append((graph.detach(), gender, gender_idx))
            y.append(shape)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = copy.deepcopy(self.x[index])
        y = copy.deepcopy(self.y[index])
        return x, y


class FTMLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cuda:0")

    def to(self, device):
        self.device = device

    def _collate_fn(self, batch):
        graphs = []
        genders = []
        gender_indexes = []
        labels = []

        for data in batch:
            x, label = data
            graph, gender, gender_index = x
            graphs.append(graph)
            genders.append(gender)
            gender_indexes.append(gender_index)
            labels.append(label)

        graphs = Batch.from_data_list(graphs).to(self.device)
        gender_indexes = torch.FloatTensor([gender_indexes]).to(self.device)
        gender_indexes = gender_indexes.view(len(batch), -1)
        labels = torch.cat(labels).to(self.device)
        return (graphs, genders, gender_indexes), labels
