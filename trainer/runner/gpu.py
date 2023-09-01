import os
import torch
from trainer.runner.base import Base
from trainer.utility.monitoring import summary_device


class SingleGPURunner(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'SingleGPURunner'

    def _check_sanity(self):
        super()._check_sanity()
        self.report_state()

        self._model.to(self.device)
        if "to" in dir(self._loader):
            self._loader.to(self.device)
            self._evaluator.to(self.device)

    def report_state(self):
        self.device = summary_device()

    def _write_log(self, **kwargs):
        properties = torch.cuda.get_device_properties(self.device)
        self._writer.add_scalars(kwargs['mode'] + '_' + properties.name + '_' + str(kwargs['epoch']),
                                 {'Memory(GB)': (properties.total_memory / (1024 ** 3)),
                                  'Loss(avg)': "%.4f" % kwargs['loss'],
                                  'Iter': kwargs['tick']})
        self._writer.flush()

    def _run_eval_epoch(self, index, progress):
        running_loss = 0.
        with torch.no_grad():
            for i, data in enumerate(progress):
                inputs, labels = data
                outputs = self._model(inputs)
                if outputs is None:
                    continue
                if not isinstance(labels, torch.Tensor):
                    loss = self._loss(outputs[0], labels[0].to(self.device))
                    for itr in range(1, len(labels)):
                        loss += self._loss(outputs[itr], labels[itr].to(self.device))
                else:
                    loss = self._loss(outputs, labels.to(self.device))
                running_loss += loss.item()

            avg_loss = running_loss / (i + 1)
            return avg_loss

    def _run_train_epoch(self, index, progress):
        running_loss = 0.
        avg_loss = 0
        mode = progress.desc
        line = "Calculating loss..."
        progress.set_description(line)
        for i, data in enumerate(progress):
            if progress.total < progress.n:
                return 1.0

            inputs, labels = data
            outputs = self._model(inputs)

            self._optimizer.zero_grad()
            if not isinstance(labels, torch.Tensor):
                loss = self._loss(outputs[0], labels[0].to(self.device))
                for itr in range(1, len(labels)):
                    loss += self._loss(outputs[itr], labels[itr].to(self.device))
            else:
                loss = self._loss(outputs, labels.to(self.device))
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()
            line = "avg_loss: %.4f, ticks: %06d" % (loss.item(), i)
            progress.set_description(line)

            if i % self._params['task']['log_interval'] == self._params['task']['log_interval'] - 1:
                save_path = os.path.join(self._params['path']['checkpoint'],
                                         self.model_name, "%08d" % index, "%08d" % i)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                # vertices, landmarks = self._model.sub_module(inputs)
                # self._viewer.save(image=inputs[0], vertices=vertices,
                #                   faces=self._model.generator.faces, text=loss.item(), save_path=save_path)
                avg_loss = running_loss / self._params['task']['log_interval']  # loss per batch
                self._write_log(epoch=index, tick=i, loss=avg_loss, mode=mode)
                self._save_model(index, i, loss=avg_loss)

                # Gather data and report
                running_loss = 0
        return avg_loss
