import copy
import os
from typing import Tuple

import cv2
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import T_co
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root: str, pre_load=False, same_output=False, resize: Tuple[int, int] = None, **kwargs):
        """
        :param root: a storage path on dataset
        :param pre_load: load images when this class is instanced
        :param same_output: If it is True, x and y is same value
        :param resize: If it is not None, return resized image
        """
        files = os.listdir(root)
        files = [os.path.join(root, x) for x in files]

        options = [transforms.ToTensor()]
        if resize is not None:
            options.append(transforms.Resize((256, 256), antialias=True))
        self.trans = transforms.Compose(options)

        self.x = []
        if pre_load:
            self.x = [self.load_image(x) for x in files]
        else:
            self.x = files

        self.pre_load = pre_load
        if same_output is True:
            self.y = []
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.pre_load:
            x = self.x[idx]
        else:
            x = self.load_image(self.x[idx])

        if len(self.y) == 0:
            y = x
        else:
            y = self.y[idx]
        return x, y

    def load_image(self, path):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.trans(image)


class ImageAgeDataset(Dataset):
    def __init__(self, pre_load=False, same_output=False, resize: Tuple[int, int] = None, **kwargs):
        """
        :param root: a storage path on dataset
        :param pre_load: load images when this class is instanced
        :param same_output: If it is True, x and y is same value
        :param resize: If it is not None, return resized image
        """
        root = r"D:\Creadto\Heritage\Dataset\CelebA\archive\img_align_celeba\img_align_celeba"
        dataframe = pd.read_csv(r"D:\Creadto\Heritage\Dataset\CelebA\archive\gender_age.csv")
        images = dataframe['filename'].to_numpy()
        ages = dataframe['age'].to_numpy()

        files = [os.path.join(root, x) for x in images]

        options = [transforms.ToTensor()]
        if resize is not None:
            options.append(transforms.Resize((256, 256), antialias=True))
        self.trans = transforms.Compose(options)

        self.x = []
        self.y = []
        for i, x in enumerate(tqdm(
                files)):
            if ages[i] < 0.1:
                continue
            self.x.append(self.load_image(x))
            y = torch.tensor(ages[i])
            self.y.append(y.unsqueeze(dim=0).type(torch.FloatTensor))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def load_image(self, path):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.trans(image)


class CelebALoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    def _collate_fn(self, batch) -> [torch.tensor, torch.tensor]:
        batch_images, batch_labels = self.make_batch(batch)
        return [batch_images.detach()], batch_labels.detach()

    def make_batch(self, batch):
        batch_images = []
        batch_labels = []

        for stub in batch:
            batch_images.append(stub[0])
            batch_labels.append(stub[1])

        batch_images = torch.stack(batch_images, dim=0).to(self.device)
        batch_labels = torch.stack(batch_labels, dim=0).to(self.device)
        return batch_images, batch_labels

    def sample(self):
        indexes = torch.randint(0, len(self.dataset), (self.batch_size, 1))
        batch = []
        for index in indexes:
            batch.append(self.dataset[index])

        batch_images, _ = self.make_batch(batch)

        result = {'images': batch_images,
                  'title': "Input image samples",
                  'device': self.device
                  }
        return result
