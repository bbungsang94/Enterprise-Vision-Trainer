import os
import pandas as pd
import torch
import json
from dataset import FLAEPDataset

def single_shot():
    # 이미지 생성하고
    # 성별 분류와 함께 랜드마크를 찍는다.
    # 20개의 shape / 10개의 expression / jaw rot 4 / neck 3
    # FLAME의 랜드마크와 비교함
    # Loss 계산
    # 입력은 이미지 로스는 랜드마크 출력은 3d
    pass


def loop():
    # Initialize
    dataset_root = r'D:\Creadto\Heritage\Dataset\GAN dataset'

    with open(os.path.join(dataset_root, 'label.txt'), "r") as f:
        labels = f.readlines()
    labels = [label.replace('\n', '') for label in labels]
    with open("./graph_info/edge_info.json", "r") as f:
        edge_info = json.load(f)
    edges = dict()
    for key, value in edge_info.items():
        edges[key] = pd.read_csv(value, header=None)
        edges[key] = edges[key].to_numpy()
        edges[key] = torch.tensor(edges[key], dtype=torch.long)

    # Dataset load
    dataset = FLAEPDataset(dataset_root=dataset_root, labels=labels, edge_info=edges)
    test = dataset[1]
    # DEA
    # Model definition
    # Dryrun model
    # train loop
    # evaluation
    # utils-display, lod, checkpoints
    pass


if __name__ == "__main__":
    loop()
