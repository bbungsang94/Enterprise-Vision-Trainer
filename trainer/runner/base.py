import copy
import os
from functools import wraps
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from trainer.utility.io import CSVWriter, make_dir, clean_folder
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
                 params: dict = None
                 ):
        self._viewer = viewer
        self._loader, self._evaluator = loaders
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._params = params

        # base
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        log_path = os.path.join(self._params['path']['log'], self._params['task']['model_name'], timestamp)
        clean_folder(os.path.join(self._params['path']['log'], self._params['task']['model_name']))
        self._writer = CSVWriter(log_path)
        self.name = 'base'
        self.model_name = self._params['task']['model_name']

        # device
        self.device = torch.device("cpu")

        # inner
        self.__loop = self.loop

    @abstractmethod
    def _run_train_epoch(self, index, progress):
        pass

    @abstractmethod
    def _run_eval_epoch(self, index, progress):
        pass

    @abstractmethod
    def report_state(self):
        pass

    @abstractmethod
    def _write_log(self, **kwargs):
        pass

    def _check_sanity(self):
        make_dir(os.path.join(self._params['path']['checkpoint'], self.model_name))
        make_dir(os.path.join(self._params['path']['log'], self.model_name))
        print(print_message(message='', line='='))
        print(print_message(message='Checking Sanity', padding=3, center=True, line='-'))
        print(print_message(message=''))
        model_input = get_input_variable_count(self._model.forward)
        loader_output = get_return_variable_count(self._loader.collate_fn)
        print(print_message(message='Loader output: ' + str(loader_output), padding=2))
        print(print_message(message='Model Input: ' + str(model_input), padding=2))
        print(print_message(message='Model output: ' + str(get_return_variable_count(self._model.forward)), padding=2))
        epoch = 0
        tick = 0
        if self._params['task']['resume']:
            print(print_message(message='Resume Process', padding=3, center=True, line='-'))
            epochs = os.listdir(os.path.join(self._params['path']['checkpoint'], self.model_name))
            if len(epochs) == 0:
                print("'\033[91m" + "Not found: can't find checkpoints of this model. Canceled the resuming" + "\033[0m")
            else:
                epochs = [int(x) for x in epochs]
                epoch = max(epochs)
                ticks = os.listdir(os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch))
                ticks = [int(x) for x in ticks]
                tick = max(ticks)
                full_path = os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch, "%08d" % tick)
                model_file = os.listdir(full_path)[0]
                self._load_model(os.path.join(full_path, model_file))
                print(print_message(message='Epoch: ' + str(epoch), padding=2))
                print(print_message(message='Tick: ' + str(tick), padding=2))
                print(print_message(message='Model Name: ' + model_file, padding=2))

        self._params['task']['itr'] = [epoch, self._params['hyperparameters']['epochs']]
        self._params['task']['tick'] = tick
        if (loader_output - 1) != model_input:
            print("'\033[91m" + "CONFLICT: Different loader output with model input shapes" + "\033[0m")

    def _save_model(self, epoch: int, tick: int, prefix=''):
        full_path = os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch, "%08d" % tick)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
        timeline = prefix + datetime.now().strftime('%Y%m%d%H%M%S') + '.pth'
        torch.save(self._model.state_dict(), os.path.join(full_path, timeline))

    def _load_model(self, full_path):
        weights = torch.load(full_path)
        self._model.load_state_dict(weights)

    def loop(self) -> None:
        self._check_sanity()
        self._viewer.show()
        outer_pbar = tqdm(range(*self._params['task']['itr']),
                          desc='Outer progress is created', position=0, leave=True)

        for epoch in range(*self._params['task']['itr']):
            if not os.path.exists(os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch)):
                os.mkdir(os.path.join(self._params['path']['checkpoint'], self.model_name, "%08d" % epoch))
            # save_option
            if epoch % self._params['task']['save_interval'] == 0:
                self._save_model(epoch, self._params['task']['tick'] + 1)
                self._viewer.save()

            # train
            self._model.train()
            train_pbar = tqdm(self._loader, desc='Train', position=0, leave=True)
            if self._params['task']['tick'] > 0:
                train_pbar.total -= copy.deepcopy(self._params['task']['tick'])
                self._params['task']['tick'] = 0
            train_mu = self._run_train_epoch(epoch, train_pbar)

            # evaluation
            self._model.eval()
            eval_pbar = tqdm(self._evaluator, desc='Eval', position=0, leave=True)
            eval_mu = self._run_eval_epoch(epoch, eval_pbar)

            # recording
            self._writer.add_scalars('Training vs Evaluation Loss',
                                     {'Training': train_mu, 'Evaluation': eval_mu, 'Epoch': epoch})
            self._writer.flush()

            line = "avg_loss(train): %.4f, avg_loss(eval): %.4f, epoch: %06d" % (train_mu, eval_mu, epoch)
            # outer_pbar.set_description(line)
            print(line)
        self._viewer.summary()
