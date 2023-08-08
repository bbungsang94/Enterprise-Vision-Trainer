from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from contents.fleap.dataset import FLAEPDataset, FLAEPDataLoader
from contents.fleap.model import BasicFLAEP
from contents.generative_model.ddpm.model import Unet


def get_dataset_fn(dataset, **kwargs) -> Dataset:
    return dataset(**kwargs)


def get_dataloader_fn(dataloader, **kwargs) -> DataLoader:
    return dataloader(**kwargs)


def get_model_fn(model, **kwargs) -> nn.Module:
    return model(**kwargs)


REGISTRY = {'FLAEPdataset': partial(get_dataset_fn, dataset=FLAEPDataset),
            'FLAEPloader': partial(get_dataloader_fn, dataloader=FLAEPDataLoader),
            'DataLoader': partial(get_dataloader_fn, dataloader=DataLoader),
            'FLAEPmodel': partial(get_model_fn, model=BasicFLAEP),
            'DDPMmodel': partial(get_model_fn, model=Unet)
            }
