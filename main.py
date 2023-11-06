import os
import torch
from torch import optim, nn
from torchvision import models, datasets
from torch.utils.data import random_split
from trainer.utility.io import read_json, ModuleLoader
from trainer.runner import REGISTRY as RUNNER
from trainer.viewer import REGISTRY as VIEWER

if __name__ == "__main__":
    config_root = './trainer/config'
    parameter = read_json(os.path.join(config_root, 'default.json'))
    module_loader = ModuleLoader(root=config_root, params=read_json(os.path.join(config_root, parameter['address'])))

    dataset = module_loader.get_module('dataset', base=datasets)

    if module_loader.params['task']['train_ratio'] == 0.0:
        train_dataset, eval_dataset = dataset.split()
    else:
        dataset_size = len(dataset)
        train_size = int(dataset_size * module_loader.params['task']['train_ratio'])
        eval_size = dataset_size - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = module_loader.get_module('loader', base=torch.utils.data, dataset=train_dataset)
    eval_loader = module_loader.get_module('loader', base=torch.utils.data, dataset=eval_dataset)
    loss = module_loader.get_module('loss', base=nn)
    model = module_loader.get_module('model', base=models)
    optimizer = module_loader.get_module('optimizer', base=optim, params=model.parameters())

    args = module_loader.get_args('viewer', module_loader.params['modules']['viewer'])
    viewer = VIEWER[module_loader.params['modules']['viewer']](**args)

    runner = RUNNER[module_loader.params['modules']['runner']](
        loaders=(train_loader, eval_loader),
        model=model,
        loss=loss,
        optimizer=optimizer,
        params=module_loader.params,
        viewer=viewer
    )

    runner.loop()
