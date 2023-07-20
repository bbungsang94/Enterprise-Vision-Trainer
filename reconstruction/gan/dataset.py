from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
from torchvision.io import read_image
import os
import torch
import cv2


class FLAEPDataset(Dataset):
    def __init__(self, dataset_root, labels, edge_info):
        self.root = dataset_root
        self.x_data = labels
        self.edges = edge_info

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        line = self.x_data[idx]
        stub = line.split(' ')
        image = read_image(os.path.join(self.root, stub[0]), )
        points = pd.read_csv(os.path.join(self.root, stub[-1]))
        points = points.to_numpy()
        x = torch.tensor([points], dtype=torch.float)