import os
import torch
from trainer.runner.base import Base
from trainer.utility.monitoring import summary_device


class FLAEPVisualize(Base):
    def _run_train_epoch(self, index, progress):
        pass

    def _run_eval_epoch(self, index, progress):
        with torch.no_grad():
            for i, data in enumerate(progress):
                inputs, labels = data
                images, graphs = inputs
                if len(images) != self._params['hyperparameters']['batch_size']:
                    continue

                vertices, landmarks = self._model.sub_module(inputs)
                outputs = self._model(inputs)

                loss = self._loss(outputs[0], labels[0].to(self.device))
                for itr in range(1, len(labels)):
                    loss += self._loss(outputs[itr], labels[itr].to(self.device))

                self._viewer.show(image=images, vertices=vertices, faces=self._model.generator.faces,
                                  text=loss.item(), save=False)

    def report_state(self):
        self.device = summary_device()

    def _write_log(self, **kwargs):
        pass
