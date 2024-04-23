import os
import json
import pandas as pd
import numpy as np
import torch


def main_body():
    from contents.regression.stm.convention import get_interactions
    from contents.regression.stm.pose import get_hands_on, get_t, get_standing, get_curve, get_sitdown

    pin_root = r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/stm/data"
    circ_root = r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/stm/data"
    minimax = pd.read_csv(r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/stm/data/minimax-val.csv", header=None, index_col=None)
    with open(os.path.join(pin_root, 'standing.json'), 'r', encoding='UTF-8-sig') as f:
        standing = json.load(f)
    with open(os.path.join(pin_root, 'sitting.json'), 'r', encoding='UTF-8-sig') as f:
        sitting = json.load(f)
    with open(os.path.join(circ_root, 'circumference.json'), 'r', encoding='UTF-8-sig') as f:
        circ_dict = json.load(f)

    minimax = minimax.to_numpy()
    minimax = {
        'max': minimax[0].astype(np.float32),
        'min': minimax[1].astype(np.float32)
    }
    dim_guide = {
        "interactions": get_interactions(),
        "landmarks": {
            "standing": standing,
            "sitting": sitting,
        },
        "circumference dict": circ_dict,
        "poses": {
            "standing": get_standing(1),
            "hands-on": get_hands_on(1),
            "t": get_t(1),
            "curve": get_curve(1),
            "sitting": get_sitdown(1),
        },
        "range": minimax
    }

    torch.save(dim_guide, "body_dimension_guide.pt")


def main_head():
    import numpy as np
    from contents.regression.ftm.convention import get_interactions

    pin_root = r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/ftm/data"
    circ_root = r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/ftm/data"
    minimax = pd.read_csv(r"D:/Creadto/Utilities/Enterprise-Vision-Trainer/contents/regression/ftm/data/minimax-val.csv", header=None, index_col=None)
    with open(os.path.join(pin_root, 'standing.json'), 'r', encoding='UTF-8-sig') as f:
        standing = json.load(f)
    with open(os.path.join(circ_root, 'circumference.json'), 'r', encoding='UTF-8-sig') as f:
        circ_dict = json.load(f)

    minimax = minimax.to_numpy()
    minimax = {
        'max': minimax[0][np.isfinite(minimax[0])].astype(np.float32),
        'min': minimax[1][np.isfinite(minimax[1])].astype(np.float32),
        'split_weight': minimax[2].astype(np.float32),
        'split_index': minimax[3].astype(np.int64)
    }
    dim_guide = {
        "interactions": get_interactions(),
        "landmarks": {
            "standard": standing,
        },
        "circumference dict": circ_dict,
        "poses": {
            "standard": torch.zeros(1, 6, dtype=torch.float32),
        },
        "range": minimax
    }
    torch.save(dim_guide, "head_dimension_guide.pt")


if __name__ == "__main__":
    main_head()
