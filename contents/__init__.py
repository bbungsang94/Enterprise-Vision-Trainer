import torch
from abc import abstractmethod, ABCMeta
from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader

from contents.basic.autoencoder.dataset import GenLoader, Gen4xLoader
from contents.basic.autoencoder.model import StaticAutoencdoer, Autoencdoer4x, DoubleEncoder
from contents.basic.facial_landmark.dataset import LandmarkDataset
from contents.basic.facial_landmark.model import BasicLandmarker
from contents.basic.unet.model import ClassicUnet
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


REGISTRY = {'FLAEP_dataset': partial(get_dataset_fn, dataset=FLAEPDataset),
            'FLAEPv2_dataset': partial(get_dataset_fn, dataset=FLAEPNoPinDataset),
            'SkinDDPM_dataset': partial(get_dataset_fn, dataset=SkinDataset),
            'Landmark_dataset': partial(get_dataset_fn, dataset=LandmarkDataset),
            'FLAEP_loader': partial(get_dataloader_fn, dataloader=FLAEPDataLoader),
            'FLAEPv2_loader': partial(get_dataloader_fn, dataloader=FLAEPNoPinLoader),
            'SkinDDPM_loader': partial(get_dataloader_fn, dataloader=SkinDataLoader),
            'DataLoader': partial(get_dataloader_fn, dataloader=DataLoader),
            'ForGen_loader': partial(get_dataloader_fn, dataloader=GenLoader),
            'ForGen4x_loader': partial(get_dataloader_fn, dataloader=Gen4xLoader),
            'FLAEPmodel': partial(get_model_fn, model=BasicFLAEP),
            'DiffusionFLAEP_model': partial(get_model_fn, model=ParamFLAEP),
            'DDPM_model': partial(get_model_fn, model=Unet),
            'SkinDDPM_model': partial(get_model_fn, model=SkinDiffusion),
            'AutoEncoder4x_model': partial(get_model_fn, model=Autoencdoer4x),
            'AutoEncoder_model': partial(get_model_fn, model=DoubleEncoder),
            'Unet_model': partial(get_model_fn, model=ClassicUnet),
            'Landmark_model': partial(get_model_fn, model=BasicLandmarker),
            'Diffusionloss': FakeLoss
            }


class TorchBase(metaclass=ABCMeta):
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
