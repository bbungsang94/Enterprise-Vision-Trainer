import os
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from trainer.utility.monitoring import get_return_variable_count, print_message, get_input_variable_count
from trainer.viewer.base import Base as Viewer
from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    def __init__(self, viewer: Optional[Viewer] = None,
                 loaders: Tuple[DataLoader, DataLoader] = None,
                 model: nn.Module = None,
                 loss: nn.Module = None,
                 optimizer: Optimizer = None,
                 params: Dict[str, ...] = None
                 ):
        self._viewer = viewer
        self._loader, self._evaluator = loaders
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._params = params

        # base
        timestamp = datetime.now().strftime('_%Y%m%d%H%M%S')
        log_path = os.path.join(self._params['path']['log'], self._params['task']['model_name'] + timestamp)
        self._writer = SummaryWriter(log_path)
        self.name = 'base'

    @abstractmethod
    def _run_single_epoch(self, index, progress):
        pass

    @abstractmethod
    def report_state(self):
        pass

    @abstractmethod
    def _write_log(self, **kwargs):
        pass

    def _check_sanity(self):
        print(print_message(message='', line='='))
        print(print_message(message='Checking Sanity', padding=3, center=True, line=''))
        print(print_message(message=''))
        loader_output = get_return_variable_count(self._loader)
        model_input = get_input_variable_count(self._model)
        print(print_message(message='Loader output: ' + str(loader_output), padding=2))
        print(print_message(message='Model Input: ' + str(model_input), padding=2))
        print(print_message(message='Model output: ' + str(get_return_variable_count(self._model)), padding=2))

        if (loader_output - 1) == model_input:
            print("CONFLICT: Different loader output with model input shapes")

    def _save_model(self, epoch: int, tick: int, prefix=''):
        full_path = os.path.join(self._params['path']['check_point'], "%08d" % epoch, "%08d" % tick)
        timeline = prefix + datetime.now().strftime('%Y%m%d%H%M%S') + '.pth'
        torch.save(self._model.state_dict(), os.path.join(full_path, timeline))

    def _load_model(self, full_path):
        weights = torch.load(full_path)
        self._model.load_state_dict(weights)

    def loop(self):
        self._check_sanity()
        self._viewer.show()

        outer_pbar = tqdm(range(self._params['hyperparameter']['epochs']), desc='Outer progress is created', position=0)

        for epoch in outer_pbar:
            # train
            self._model.train()
            train_pbar = tqdm(self._loader, desc='Inner progress is created', position=1, leave=False)
            train_mu = self._run_single_epoch(epoch, train_pbar)

            # evaluation
            self._model.eval()
            eval_pbar = tqdm(self._evaluator, desc='Inner progress is created', position=2, leave=False)
            eval_mu = self._run_single_epoch(epoch, eval_pbar)

            # recording
            self._writer.add_scalars('Training vs. Evaluation Loss',
                                     {'Training': train_mu, 'Evaluation': eval_mu},
                                     epoch)
            self._writer.flush()

            # save_option
            if epoch % self._params['task']['save_interval']:
                self._save_model(epoch + 1, 0)