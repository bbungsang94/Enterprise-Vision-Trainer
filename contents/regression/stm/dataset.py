import os

import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import T_co
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class STMWrapper:
    def __init__(self, train_root, test_path):
        self.train = STMTrainSet(dataset_root=train_root)
        #self.test = STMTestSet(dataset_path=test_path)
        self.test = STMEvalSet(dataset_root=train_root.replace('parameter', 'eval'))

    def split(self):
        return self.train, self.test


class STMTrainSet(Dataset):
    def __init__(self, dataset_root):
        files = os.listdir(dataset_root)
        genders = []
        x = []
        label = []
        pbar = tqdm(files, desc="Loading train dataset")
        for file in pbar:
            pth = torch.load(os.path.join(dataset_root, file))
            genders.append(pth['input']['gender'])
            measure = torch.FloatTensor(pth['output']['measure'])
            if pth['input']['gender'] == "male":
                measure[5] = 0.0
                measure[29] = 0.0
            label.append(pth['input']['shape'])
            x.append(measure)

        self.x_data = torch.stack(x)
        self.gender = genders
        self.y_data = torch.concat(label).to(torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.gender[idx]), self.y_data[idx]


class STMEvalSet(Dataset):
    def __init__(self, dataset_root):
        files = os.listdir(dataset_root)
        genders = []
        x = []
        label = []
        pbar = tqdm(files, desc="Loading eval dataset")
        for file in pbar:
            pth = torch.load(os.path.join(dataset_root, file))
            genders.append(pth['input']['gender'])
            measure = torch.FloatTensor(pth['output']['measure'])
            if pth['input']['gender'] == "male":
                measure[5] = 0.0
                measure[29] = 0.0
            label.append(measure.detach())
            x.append(measure.detach())

        self.x_data = torch.stack(x)
        self.gender = genders
        self.y_data = torch.stack(label)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.gender[idx]), self.y_data[idx]


class STMTestSet(Dataset):
    def __init__(self, dataset_path):
        dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
        dataset.columns = [x.replace(' ', '') for x in dataset.columns.to_list()]
        self.columns = ["머리위로뻗은주먹높이", "키", "목뒤높이", "어깨높이", "겨드랑높이", "허리기준선높이(여)", "허리높이",
                        "위앞엉덩뼈가시높이", "샅높이", "가쪽복사높이", "주먹높이", "팔꿈치높이(팔굽힌)", "가슴너비", "허리너비",
                        "엉덩이너비", "종아리아래너비(발목너비)", "겨드랑두께", "가슴두께", "허리두께", "목둘레", "목밑뒤길이",
                        "목밑둘레", "겨드랑둘레", "편위팔둘레", "편팔꿈치둘레", "손목둘레", "위팔둘레(팔굽힌)", "가슴둘레",
                        "젖가슴둘레", "젖가슴아래둘레(여)", "허리둘레", "배꼽수준허리둘레", "엉덩이둘레", "넙다리둘레", "무릎둘레",
                        "장딴지둘레", "목옆뒤허리둘레선길이", "목옆젖꼭지길이", "목옆젖꼭지허리둘레선길이", "샅앞뒤길이",
                        "어깨목뒤길이(왼)", "목뒤어깨사이길이", "위팔길이", "팔길이", "앉은키", "앉은배두께",
                        "팔꿈치주먹수평길이(팔굽힌)"]
        kor_eng = {'남': "male", '여': "female"}
        temp = dataset["성별"]
        self.gender = [kor_eng[x] for x in temp.to_list()]
        self.y_data = dataset[self.columns].to_numpy()
        self.y_data = np.nan_to_num(self.y_data)
        self.y_data = torch.FloatTensor(self.y_data)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return (self.y_data[idx], self.gender[idx]), self.y_data[idx]


class STMLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    def _collate_fn(self, batch) -> [torch.tensor, torch.tensor]:
        tup = self.make_batch(batch)
        return tup

    def make_batch(self, batch):
        measure = []
        gender = []
        labels = []

        for stub, label in batch:
            measure.append(stub[0])
            gender.append(stub[1])
            labels.append(label)

        measure = torch.stack(measure).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return (measure, gender), labels
