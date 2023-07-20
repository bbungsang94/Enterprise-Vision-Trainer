import os
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
    dataset_root = r'D:\Creadto\Heritage\Dataset\GAN dataset'
    # Dataset load
    with open(os.path.join(dataset_root, 'label.txt'), "r") as f:
        labels = f.readlines()
    labels = [label.replace('\n', '') for label in labels]
    dataset = FLAEPDataset(labels=labels)
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
