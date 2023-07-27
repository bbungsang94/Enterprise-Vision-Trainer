import torch
from trainer.runner.base import Base
from trainer.utility.monitoring import summary_device


class SingleGPURunner(Base):
    def __init__(self):
        super().__init__()
        self.name = 'SingleGPURunner'
        self.device = None

    def _check_sanity(self):
        super()._check_sanity()
        self.report_state()
        # GPU로 옮기는 작업이 자동으로 이루어짐 만약 CPU라면 스위치가 따로 있어야함
        super()._model.to(self.device)
        super()._loader.to(self.device)
        super()._evaluator.to(self.device)

    def report_state(self):
        self.device = summary_device()

    def _write_log(self, **kwargs):
        properties = torch.cuda.get_device_properties(self.device)
        self._writer.add_scalars(properties.name + '/' + str(kwargs['epoch']),
                                 {'Memory(GB)': (properties.total_memory / (1024 ** 3)),
                                  'Loss(avg)': "%.4f" % kwargs['loss']},
                                 kwargs['tick'])
        self._writer.flush()

    def _run_single_epoch(self, index, progress):
        running_loss = 0.
        avg_loss = 0
        for i, data in enumerate(progress):
            inputs, labels = data

            super()._optimizer.zero_grad()
            outputs = super()._model(inputs)
            loss = super()._loss(outputs, labels)
            loss.backward()
            super()._optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % super()._params['task']['log_interval'] == 0:
                avg_loss = running_loss / super()._params['task']['log_interval']  # loss per batch
                self._write_log(epoch=index, tick=i, loss=avg_loss)

        return avg_loss
