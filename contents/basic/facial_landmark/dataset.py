import os
import cv2
import torch
import pandas as pd
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class LandmarkDataset(Dataset):
    def __init__(self, dataset_root, input_folder, label_folder, prefix='68-'):
        self.root = dataset_root
        self.input_data = os.listdir(os.path.join(self.root, input_folder))
        self.label_data = os.listdir(os.path.join(self.root, label_folder))

        self.input_data = [os.path.join(self.root, input_folder, x) for x in self.input_data]
        self.label_data = [os.path.join(self.root, label_folder, x) for x in self.label_data if prefix in x]
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ])
        self.resizer = torchvision.transforms.Resize((256, 256), antialias=True)
        self.x_data = []
        self.shapes = []
        self.y_data = []
        
        # 예상되는 램 사이즈를 보고 sanity check해야함
        for idx in tqdm(range(len(self.input_data) // 2), desc="Loading LandmarkDataset dataset from " + input_folder):
            image = cv2.imread(self.input_data[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # mean, std = image.mean(), image.std()
            # normalize = torchvision.transforms.Normalize(mean, std)
            # image = normalize(image)
            image = trans(image)

            label = pd.read_csv(self.label_data[idx], header=None)
            label = label.to_numpy()
            label = torch.from_numpy(label).type(torch.FloatTensor)

            self.x_data.append(image)
            self.y_data.append(label[:, :3])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
