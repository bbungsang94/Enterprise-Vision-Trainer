import os
import torch
from torch_geometric.data import Batch
from torch.utils.data.dataloader import T_co
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class FTMGraphset(Dataset):
    def __init__(self, dataset_root, node_shaking=False):
        files = os.listdir(dataset_root)
        x = []
        pbar = tqdm(files, desc="Loading train dataset")
        for file in pbar:
            pth = torch.load(os.path.join(dataset_root, file))
            x.append(pth['output']['graph'])

        self.x_data = x

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        graph = self.x_data[idx]
        edge_attributes = graph.edge_attr
        return self.x_data[idx]


class FTMGraphLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], **kwargs):
        del kwargs['name']
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    def _collate_fn(self, batch) -> [Batch]:
        batch = Batch.from_data_list(batch)
        batch = batch.to(self.device)
        return (batch.x, batch.edge_index), (batch.x, batch.edge_attr)
