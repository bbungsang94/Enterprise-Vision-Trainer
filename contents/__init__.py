import torch
from abc import abstractmethod, ABCMeta
from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from contents.reconstruction.fleap.dataset import FLAEPDataset, FLAEPDataLoader, FLAEPNoPinDataset, FLAEPNoPinLoader
from contents.reconstruction.fleap.model import BasicFLAEP, DiffuseFLAEP
from contents.generative_model.ddpm.model import Unet


def get_dataset_fn(dataset, **kwargs) -> Dataset:
    return dataset(**kwargs)


def get_dataloader_fn(dataloader, **kwargs) -> DataLoader:
    return dataloader(**kwargs)


def get_model_fn(model, **kwargs) -> nn.Module:
    return model(**kwargs)


REGISTRY = {'FLAEPdataset': partial(get_dataset_fn, dataset=FLAEPDataset),
            'FLAEPloader': partial(get_dataloader_fn, dataloader=FLAEPDataLoader),
            'FLAEPv2dataset': partial(get_dataset_fn, dataset=FLAEPNoPinDataset),
            'FLAEPv2loader': partial(get_dataloader_fn, dataloader=FLAEPNoPinLoader),
            'DataLoader': partial(get_dataloader_fn, dataloader=DataLoader),
            'FLAEPmodel': partial(get_model_fn, model=BasicFLAEP),
            'DiffusionFLAEPmodel': partial(get_model_fn, model=DiffuseFLAEP),
            'DDPMmodel': partial(get_model_fn, model=Unet)
            }


class Base(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def forward(self, x) -> [torch.tensor]:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def state_dict(self, **kwargs):
        pass

    @abstractmethod
    def load_state_dict(self, weights):
        pass
