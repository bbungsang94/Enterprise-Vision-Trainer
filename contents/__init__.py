from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from contents.fleap.dataset import FLAEPDataset, FLAEPDataLoader, FLAEPNoPinDataset, FLAEPNoPinLoader
from contents.fleap.model import BasicFLAEP, DiffuseFLAEP
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
