import open3d as o3d
import torch

from contents.reconstruction.fleap.model import FLAMESet
from contents.reconstruction.threeDMM.flame import get_parser, FLAME


class FLAMERenderer:
    def __init__(self, params):
        self.generator = FLAMESet(**params)

    def render(self, gender, param, **kwargs):
        proc_itr = 1
        if 'dual' in kwargs:
            proc_itr = 2 if kwargs['dual'] else 1

        if len(param) != self.generator.batch_size:
            return None

        for _ in range(proc_itr):
            model = self.generator(genders=gender, **param)

    def