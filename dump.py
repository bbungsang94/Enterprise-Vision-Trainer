import os
import json

import torch

from contents.regression.stm.convention import get_interactions
from contents.regression.stm.pose import get_hands_on, get_t, get_standing, get_curve, get_sitdown


def main():
    pin_root = r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/stm/data"
    circ_root = r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/stm/data"
    with open(os.path.join(pin_root, 'standing.json'), 'r', encoding='UTF-8-sig') as f:
        standing = json.load(f)
    with open(os.path.join(pin_root, 'sitting.json'), 'r', encoding='UTF-8-sig') as f:
        sitting = json.load(f)
    with open(os.path.join(circ_root, 'circumference.json'), 'r', encoding='UTF-8-sig') as f:
        circ_dict = json.load(f)
    dim_guide = {
        "interactions": get_interactions(),
        "landmarks":{
            "sitting": sitting,
            "standing": standing,
        },
        "circumference dict": circ_dict,
        "poses": {
            "hands-on": get_hands_on(1),
            "standing": get_standing(1),
            "t": get_t(1),
            "curve": get_curve(1),
            "sitting": get_sitdown(1)
        }
    }
    torch.save(dim_guide, "dimension_guide.pt")

if __name__ == "__main__":
    main()