import os
from datetime import datetime
from typing import Optional, Dict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from trainer.viewer.base import Base as Viewer
from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    def __init__(self, viewer: Optional[Viewer] = None,
                 loader: DataLoader = None,
                 model: nn.Module = None,
                 loss: nn.Module = None,
                 optimizer: Optimizer = None,
                 params: Dict[str, ...] = None
                 ):
        self._viewer = viewer
        self._loader = loader
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._params = params

        # base
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        self.name = 'base'

    @abstractmethod
    def run_single_epoch(self):
        pass

    @abstractmethod
    def activate_sanity(self):
        pass

    def save_model(self, epoch: int, tick: int, prefix=''):
        full_path = os.path.join(self._params['path']['check_point'], "%08d" % epoch, "%08d" % tick)
        timeline = prefix + datetime.now().strftime('%Y%m%d%H%M%S') + '.pth'
        torch.save(self._model.state_dict(), os.path.join(full_path, timeline))

    def load_model(self, full_path):
        weights = torch.load(full_path)
        self._model.load_state_dict(weights)
