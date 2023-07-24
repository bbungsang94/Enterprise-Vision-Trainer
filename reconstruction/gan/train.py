import copy
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


def fit_output(model_output, flame_output, aligner, vis=False):
    for b in range(model_output.shape[0]):
        label_landmark = model_output[b].detach().cpu().numpy()
        model_landmark = flame_output[1][b].detach().cpu().numpy()

        label_min = label_landmark.min(axis=0)
        model_min = model_landmark.min(axis=0)
        label_max = label_landmark.max(axis=0)
        model_max = model_landmark.max(axis=0)
        label_gap = label_max - label_min
        model_gap = model_max - model_min

        model_landmark = model_max - model_landmark

        # # 0 - 1 scaling
        # label_landmark = 2 * ((label_landmark - label_min) / label_gap)
        # model_landmark = 2 * ((model_landmark - model_min) / model_gap)
        #
        # model_landmark = 2 - model_landmark
        # label_landmark = label_landmark - 1
        # model_landmark = model_landmark - 1

        # 점들을 scatter plot으로 시각화
        aligner.update_points(label_landmark, 'landmark')
        aligner.align_seq()
        label_landmark = aligner.offset_seq()

        aligner.update_points(model_landmark, 'points68')
        aligner.align_seq()
        model_landmark = aligner.offset_seq()
        if vis:
            # PointCloud 시각화
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(label_landmark[:, 0], label_landmark[:, 1], label_landmark[:, 2], c='r', marker='o', s=10)
            ax.scatter(model_landmark[:, 0], model_landmark[:, 1], model_landmark[:, 2], c='b', marker='o',
                       s=10)
            # 축 범위 설정 (필요에 따라 적절하게 조정)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()

        return model_landmark, label_landmark


def draw_pinbox(output, truth, pin_boxes):
    import open3d as o3d
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(output)
    pcd1.paint_uniform_color([0, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(truth)
    pcd2.paint_uniform_color([0, 0, 0])
    pcd2 = pcd2.translate((0, 1, 0))

    align_box = pin_boxes['Align']
    eyes_box = pin_boxes['Eyes']
    nose_box = pin_boxes['Nose']
    lip_box = pin_boxes['Lip']
    eyebrow_box = pin_boxes['Eyebrow']

    o3d.visualization.draw_geometries([pcd1, pcd2])



def FLAEP_loss(output, truth, pin_boxes):
    draw_pinbox(output, truth, pin_boxes)
    modes = {'landmark': truth, 'points68': output}
    losses = dict()
    for mode, points in modes.items():
        loss = dict()
        for key, box in pin_boxes.items():
            box.switch_mode(mode)
            pin_boxes[key] = box

        eyes_box = pin_boxes['Eyes']
        nose_box = pin_boxes['Nose']
        lip_box = pin_boxes['Lip']
        eyebrow_box = pin_boxes['Eyebrow']

        # 코 중심, 턱 끝
        nose_tip = points[nose_box()[0]]
        jaw_tip = points[pin_boxes['Align']()[1]]
        # 눈, 입 정보
        eyes_points = points[eyes_box()]
        lip_points = points[lip_box()]
        eyebrow_points = points[eyebrow_box()]

        # 코 중심으로부터의 턱 아래의 거리 - 입이 벌려진 높이
        lip_height = np.linalg.norm(lip_points[2] - lip_points[3])
        base = np.linalg.norm(jaw_tip - nose_tip) - lip_height

        # 입의 벌린 정도, 입의 길이
        loss['Lip_C_H'] = lip_height / base
        lip_width = np.linalg.norm(lip_points[0] - lip_points[1])
        loss['Lip_C_W'] = lip_width / base

        # 눈의 높이(2, 3), 길이(0, 1)
        l_e_width = np.linalg.norm(eyes_points[1] - eyes_points[0])
        loss['Eye_L_W'] = l_e_width / base
        l_e_height = np.linalg.norm(eyes_points[2] - eyes_points[3])
        loss['Eye_L_H'] = l_e_height / base
        r_e_width = np.linalg.norm(eyes_points[5] - eyes_points[4])
        loss['Eye_R_W'] = r_e_width / base
        r_e_height = np.linalg.norm(eyes_points[6] - eyes_points[7])
        loss['Eye_R_H'] = r_e_height / base

        # 코 중심으로부터의 눈가 거리
        dist = np.linalg.norm(eyes_points[0] - nose_tip)
        loss['Eye_L_I'] = dist / base
        dist = np.linalg.norm(eyes_points[1] - nose_tip)
        loss['Eye_L_O'] = dist / base
        dist = np.linalg.norm(eyes_points[4] - nose_tip)
        loss['Eye_R_I'] = dist / base
        dist = np.linalg.norm(eyes_points[5] - nose_tip)
        loss['Eye_R_O'] = dist / base

        # 코 중심으로부터의 입 양 끝 거리
        dist = np.linalg.norm(lip_points[0] - nose_tip)
        loss['Lip_L_I'] = dist / base
        dist = np.linalg.norm(lip_points[1] - nose_tip)
        loss['Lip_R_O'] = dist / base
        # 코 중심으로부터의 눈썹의 거리
        dist = np.linalg.norm(eyebrow_points[1] - nose_tip)
        loss['EB_L_I'] = dist / base
        dist = np.linalg.norm(eyebrow_points[3] - nose_tip)
        loss['EB_L_O'] = dist / base
        dist = np.linalg.norm(eyebrow_points[5] - nose_tip)
        loss['EB_R_I'] = dist / base
        dist = np.linalg.norm(eyebrow_points[7] - nose_tip)
        loss['EB_R_O'] = dist / base
        losses[mode] = copy.deepcopy(loss)
    return losses


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
    y_pred, y_true = fit_output(landmarks, (vertices, landmark), pre_loss, True)
    losses = FLAEP_loss(y_pred, y_true, pin_boxes)
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
