import os
import pandas as pd
import torch
import json
import numpy as np
import pyrender
import trimesh
from models.flame import FLAME, get_parser
from utility.monitoring import summary_device
from dataset import FLAEPDataset, FLAEPDataLoader
from models.flaep import FLAEP
from reconstruction.pinning.pins.pin import PinLoader
from facial_landmarks.cv_mesh.align import Aligner
import matplotlib.pyplot as plt


def single_shot():
    # 이미지 생성하고
    # 성별 분류와 함께 랜드마크를 찍는다.
    # 20개의 shape / 10개의 expression / jaw rot 4 / neck 3
    # FLAME의 랜드마크와 비교함
    # Loss 계산
    # 입력은 이미지 로스는 랜드마크 출력은 3d
    pass


def visualize_output(vertices, faces, landmark):
    for i in range(vertices.shape[0]):
        vertex = vertices[i].detach().cpu().numpy().squeeze()
        joints = landmark[i].detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([vertex.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        tri_mesh = trimesh.Trimesh(vertex, faces, vertex_colors=vertex_colors)
        tri_mesh.export('./sample.obj', file_type='obj')
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True)


def debug_loss(model_output, flame_output, aligner):
    for b in range(model_output.shape[0]):
        label_landmark = model_output[b].detach().cpu().numpy()
        model_landmark = flame_output[1][b].detach().cpu().numpy()

        # 점들을 scatter plot으로 시각화
        aligner.update_points(label_landmark)
        aligner.align_seq()
        label_landmark = aligner.flip_seq()

        label_min = label_landmark.min(axis=0)
        model_min = model_landmark.min(axis=0)
        label_max= label_landmark.max(axis=0)
        model_max = model_landmark.max(axis=0)
        label_gap = label_max - label_min
        model_gap = model_max - model_min

        # 0 - 1 scaling
        label_landmark = (label_landmark - label_min) / label_gap
        model_landmark = (model_landmark - model_min) / model_gap

        # PointCloud 시각화
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(label_landmark[:, 0], label_landmark[:, 1], label_landmark[:, 2], c='r', marker='o', s=10)
        ax.scatter(model_landmark[:, 0], 1.0 - model_landmark[:, 1], 1.0 - model_landmark[:, 2], c='b', marker='o',
                   s=10)
        # 축 범위 설정 (필요에 따라 적절하게 조정)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


def loop(batch_size=4, epochs=300, learning_rate=1e-3):
    # Initialize
    dataset_root = r'D:\Creadto\Heritage\Dataset\GAN dataset'

    with open(os.path.join(dataset_root, 'label.txt'), "r") as f:
        labels = f.readlines()
    labels = [label.replace('\n', '') for label in labels]
    with open("./graph_info/edge_info.json", "r") as f:
        edge_info = json.load(f)
    edges = dict()
    for key, value in edge_info.items():
        edge = pd.read_csv(value, header=None)
        edges[key] = edge.to_numpy()
    pin_boxes = PinLoader.load_pins(path='../pinning/pins', filename='pin_info.json')

    # Dataset load
    dataset = FLAEPDataset(dataset_root=dataset_root, labels=labels, edge_info=edges)
    loader = FLAEPDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # DEA
    # Model definition
    gen_args = get_parser()
    gen_args.batch_size = batch_size
    generator = FLAME(config=gen_args)
    model = FLAEP()
    pre_loss = Aligner(pin_boxes=pin_boxes)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # Dryrun model
    device = summary_device()
    model.to(device)
    generator.to(device)

    images, graphs, landmarks = next(iter(loader))
    output = model(images, graphs)
    vertices, landmark = generator(**output)
    # visualize_output(vertices=vertices, landmark=landmark, faces=generator.faces)
    debug_loss(landmarks, (vertices, landmark), pre_loss)
    print(output)
    # train loop
    best_accuracy = 0.0
    for tick in range(epochs):
        running_loss = dict()
        last_loss = dict()
        for i, (images, graphs, landmarks) in enumerate(loader):
            output = model(images, graphs)
            vertices, landmark = generator(**output)
            pass
        optimizer.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    # evaluation
    # utils-display, lod, checkpoints
    pass


if __name__ == "__main__":
    loop()
