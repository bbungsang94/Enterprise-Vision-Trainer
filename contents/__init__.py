import torch
from abc import abstractmethod, ABCMeta
from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader

from contents.autoencoder.dataset import FTMGraphset, FTMGraphLoader
from contents.autoencoder.graph import GCNAutoencoder
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
from contents.regression.ftm.dataset import FTMDataset, FTMLoader
from contents.regression.ftm.model import FTMRegression
from contents.regression.stm.model import STMRegression
from contents.regression.stm.dataset import STMWrapper, STMLoader


def get_dataset_fn(dataset, **kwargs) -> Dataset:
    return dataset(**kwargs)


def get_dataloader_fn(dataloader, **kwargs) -> DataLoader:
    return dataloader(**kwargs)


def get_model_fn(model, **kwargs) -> nn.Module:
    return model(**kwargs)


class Contents:
    def __init__(self):
        datasets = {
            'FLAEP': partial(get_dataset_fn, dataset=FLAEPDataset),
            'FLAEPv2': partial(get_dataset_fn, dataset=FLAEPNoPinDataset),
            'SkinDDPM': partial(get_dataset_fn, dataset=SkinDataset),
            'Landmark': partial(get_dataset_fn, dataset=LandmarkDataset),
            'STM': partial(get_dataset_fn, dataset=STMWrapper),
            'GraphFTM': partial(get_dataset_fn, dataset=FTMGraphset),
            'FTM': partial(get_dataset_fn, dataset=FTMDataset),
        }
        loaders = {
            'FLAEP': partial(get_dataloader_fn, dataloader=FLAEPDataLoader),
            'FLAEPv2': partial(get_dataloader_fn, dataloader=FLAEPNoPinLoader),
            'SkinDDPM': partial(get_dataloader_fn, dataloader=SkinDataLoader),
            'ForGen': partial(get_dataloader_fn, dataloader=GenLoader),
            'ForGen4x': partial(get_dataloader_fn, dataloader=Gen4xLoader),
            'STM': partial(get_dataloader_fn, dataloader=STMLoader),
            'GraphLoader': partial(get_dataloader_fn, dataloader=FTMGraphLoader),
            'FTM': partial(get_dataloader_fn, dataloader=FTMLoader)
        }
        models = {
            'FLAEP': partial(get_model_fn, model=BasicFLAEP),
            'DiffusionFLAEP': partial(get_model_fn, model=ParamFLAEP),
            'DDPM': partial(get_model_fn, model=Unet),
            'SkinDDPM': partial(get_model_fn, model=SkinDiffusion),
            'AutoEncoder4x': partial(get_model_fn, model=Autoencdoer4x),
            'AutoEncoder': partial(get_model_fn, model=DoubleEncoder),
            'Unet': partial(get_model_fn, model=ClassicUnet),
            'Landmark': partial(get_model_fn, model=BasicLandmarker),
            'STM': partial(get_model_fn, model=STMRegression),
            'GCNAutoencoder': partial(get_model_fn, model=GCNAutoencoder),
            'FTM': partial(get_model_fn, model=FTMRegression),
        }
        losses = {
            'Diffusion': FakeLoss
        }
        optimizers = {

        }

        self.motherboard = {
            'dataset': datasets,
            'loader': loaders,
            'model': models,
            'loss': losses,
            'optimizer': optimizers
        }

    def __getitem__(self, key):
        result = None
        module = key.split('_')
        if module[0] in self.motherboard[module[-1]]:
            result = self.motherboard[module[-1]][module[0]]
        return result


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
