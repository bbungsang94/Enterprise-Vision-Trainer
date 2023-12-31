from functools import partial

from trainer.runner.cpu import SingleCPURunner
from trainer.runner.gpu import SingleGPURunner
from trainer.runner.visualize import FLAEPVisualize
from trainer.runner.base import Base


def get_runner_fn(runner, **kwargs) -> Base:
    return runner(**kwargs)


REGISTRY = {'SingleGPU': partial(get_runner_fn, runner=SingleGPURunner),
            'CPU': partial(get_runner_fn, runner=SingleCPURunner),
            'FLAEPVisualize': partial(get_runner_fn, runner=FLAEPVisualize)
            }
