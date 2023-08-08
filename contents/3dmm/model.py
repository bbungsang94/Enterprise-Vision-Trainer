from typing import Dict, List

import torch


class MorphableModel:
    def __init__(self, params: Dict[str, ...]):
        self.params = params
        self.constraints: Dict[str, int] = self.params['constraints']
        self._image_size = self.params['image_size']
        pass

    def get_index(self, key: str, constraints: Dict[str, int] = None) -> int:
        if not constraints:
            constraints = self.constraints
        idx = 0
        for k, v in constraints.items():
            if k != key:
                idx += v
            else:
                break
        return idx

    def readjust_to_input(self, pred_3dmm: torch.Tensor, paddings: List[int], scale: float) -> torch.Tensor:
        scale_idx = self.get_index("scale")
        translation_idx = self.get_index("translation")

        old_flame_params_scale = pred_3dmm[:, scale_idx: scale_idx + self.constraints["scale"]]
        old_flame_params_translation = pred_3dmm[
                                       :, translation_idx: translation_idx + self.constraints["translation"]
                                       ]

        new_flame_params_scale = (old_flame_params_scale + 1.0) / scale - 1.0
        new_flame_params_translation = (
                                               old_flame_params_translation + 1.0 - torch.Tensor(
                                           [[paddings[2], paddings[0], 0]]) * 2 / self._image_size
                                       ) / scale - 1.0

        pred_3dmm[:, scale_idx: scale_idx + self.constraints["scale"]] = \
            new_flame_params_scale
        pred_3dmm[:, translation_idx: translation_idx + self.constraints["translation"]] = \
            new_flame_params_translation

        return pred_3dmm
