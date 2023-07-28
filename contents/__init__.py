from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from contents.reconstruction.gan.dataset import FLAEPDataset, FLAEPDataLoader
from contents.reconstruction.gan.models.flaep import FLAEP


def get_dataset_fn(dataset, **kwargs) -> Dataset:
    return dataset(**kwargs)


def get_dataloader_fn(dataloader, **kwargs) -> DataLoader:
    return dataloader(**kwargs)


def get_model_fn(model, **kwargs) -> nn.Module:
    return model(**kwargs)


REGISTRY = {'FLAEPdataset': partial(get_dataset_fn, dataset=FLAEPDataset),
            'FLAEPloader': partial(get_dataloader_fn, dataloader=FLAEPDataLoader),
            'DataLoader': partial(get_dataloader_fn, dataloader=DataLoader),
            'FLAEPmodel': partial(get_model_fn, model=FLAEP)
            }
