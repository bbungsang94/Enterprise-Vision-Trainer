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

def debug_loss(model_output, flame_output):
    for b in range(model_output.shape[0]):
        label_landmark = model_output[b].detach()
        model_landmark = flame_output[1][b].detach()

        label_min, _ = label_landmark.min(dim=0)
        model_min, _ = model_landmark.min(dim=0)

        label_max, _ = label_landmark.max(dim=0)
        model_max, _ = model_landmark.max(dim=0)

        label_gap = label_max - label_min
        model_gap = model_max - model_min
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

    # Dataset load
    dataset = FLAEPDataset(dataset_root=dataset_root, labels=labels, edge_info=edges)
    loader = FLAEPDataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # DEA
    # Model definition
    gen_args = get_parser()
    gen_args.batch_size = batch_size
    generator = FLAME(config=gen_args)
    model = FLAEP()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # Dryrun model
    device = summary_device()
    model.to(device)
    generator.to(device)

    images, graphs, landmarks = next(iter(loader))
    output = model(images, graphs)
    vertices, landmark = generator(**output)
    #visualize_output(vertices=vertices, landmark=landmark, faces=generator.faces)
    debug_loss(landmarks, (vertices, landmark))
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
