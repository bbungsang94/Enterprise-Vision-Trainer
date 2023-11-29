import os
import torch
from torch import optim, nn
from torchvision import models, datasets
from torch.utils.data import random_split
from trainer.utility.io import read_json, ModuleLoader
from trainer.runner import REGISTRY as RUNNER
from trainer.viewer import REGISTRY as VIEWER


def get_loader(root='./trainer/config'):
    parameter = read_json(os.path.join(root, 'default.json'))
    module_loader = ModuleLoader(root=root, params=read_json(os.path.join(root, 'contents', parameter['address'])))
    return module_loader


def get_dataloaders(loader: ModuleLoader):
    dataset = loader.get_module('dataset', base=datasets)

    if loader.params['task']['train_ratio'] == 0.0:
        train_dataset, eval_dataset = dataset.split()
    else:
        dataset_size = len(dataset)
        train_size = int(dataset_size * loader.params['task']['train_ratio'])
        eval_size = dataset_size - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = loader.get_module('loader', base=torch.utils.data, dataset=train_dataset)
    eval_loader = loader.get_module('loader', base=torch.utils.data, dataset=eval_dataset)
    return train_loader, eval_loader


def run():
    loader = get_loader()

    args = loader.get_args('viewer', loader.params['modules']['viewer'])
    viewer = VIEWER[loader.params['modules']['viewer']](**args)
    model = loader.get_module('model', base=models)

    kwargs = {
        'loaders': get_dataloaders(loader),
        'model': model,
        'loss': loader.get_module('loss', base=nn),
        'metric': loader.get_module('metric', base=nn),
        'optimizer': loader.get_module('optimizer', base=optim, params=model.parameters()),
        'viewer': viewer,
        'params': loader.params,
    }
    runner = RUNNER[loader.params['modules']['runner']](**kwargs)
    runner.loop()


if __name__ == "__main__":
    run()
