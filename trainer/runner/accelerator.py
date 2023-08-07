import torch
from accelerate import Accelerator
from ema_pytorch import EMA

from modules.utils import cycle
from trainer.runner.base import Base
from trainer.utility.monitoring import summary_device


class AcceleratorRunner(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        split_batches = self._params['hyperparameters']['split_batches']
        mixed_precision_type = self._params['hyperparameters']['mixed_precision_type']
        amp = self._params['hyperparameters']['amp']
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )
        if self.accelerator.is_main_process:
            ema_decay = self._params['hyperparameters']['ema_decay']
            ema_update_every = self._params['hyperparameters']['ema_update_every']
            self.ema = EMA(self._model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        loader, evaluator = self.accelerator.prepare(self._loader, self._evaluator)
        self._loader = cycle(loader)
        self._evaluator = cycle(evaluator)
        self._model, self._optimizer = self.accelerator.prepare(self._model, self._optimizer)

        calculate_fid = self._params['hyperparameters']['calculate_fid']
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

    def _run_train_epoch(self, index, progress):
        pass

    def _run_eval_epoch(self, index, progress):
        pass

    def report_state(self):
        self.device = summary_device()

    def _write_log(self, **kwargs):
        pass
