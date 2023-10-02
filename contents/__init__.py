import torch
from abc import abstractmethod, ABCMeta
from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from contents.reconstruction.fleap.dataset import FLAEPDataset, FLAEPDataLoader, FLAEPNoPinDataset, FLAEPNoPinLoader
from contents.reconstruction.fleap.model import BasicFLAEP, ParamFLAEP
from contents.generative_model.ddpm.model import Unet
from contents.generative_model.skinpatch.dataset import SkinDataset, SkinDataLoader
from contents.generative_model.skinpatch.model import SkinDiffusion
from contents.generative_model.skinpatch.loss import FakeLoss


def get_dataset_fn(dataset, **kwargs) -> Dataset:
    return dataset(**kwargs)


def get_dataloader_fn(dataloader, **kwargs) -> DataLoader:
    return dataloader(**kwargs)


def get_model_fn(model, **kwargs) -> nn.Module:
    return model(**kwargs)


REGISTRY = {'FLAEPdataset': partial(get_dataset_fn, dataset=FLAEPDataset),
            'FLAEPv2dataset': partial(get_dataset_fn, dataset=FLAEPNoPinDataset),
            'SkinDDPMdataset': partial(get_dataset_fn, dataset=SkinDataset),
            'FLAEPloader': partial(get_dataloader_fn, dataloader=FLAEPDataLoader),
            'FLAEPv2loader': partial(get_dataloader_fn, dataloader=FLAEPNoPinLoader),
            'SkinDDPMloader': partial(get_dataloader_fn, dataloader=SkinDataLoader),
            'DataLoader': partial(get_dataloader_fn, dataloader=DataLoader),
            'FLAEPmodel': partial(get_model_fn, model=BasicFLAEP),
            'DiffusionFLAEPmodel': partial(get_model_fn, model=ParamFLAEP),
            'DDPMmodel': partial(get_model_fn, model=Unet),
            'SkinDDPMmodel': partial(get_model_fn, model=SkinDiffusion),
            'Diffusionloss': FakeLoss
            }


class Base(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.device = torch.device("cpu")

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
