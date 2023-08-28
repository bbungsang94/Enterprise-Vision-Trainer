import copy
import os
import pandas as pd
import torch
import json
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm
from datetime import datetime
from models.flame import FLAME, get_parser
from utility.monitoring import summary_device
from contents.reconstruction.fleap.dataset import FLAEPDataset, FLAEPDataLoader
from contents.reconstruction.fleap.model import FLAEPv1
from contents import PinLoader
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


def draw_pinbox(output, truth, pin_boxes):
    import open3d as o3d
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(output)
    pcd1.paint_uniform_color([0, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(truth)
    pcd2.paint_uniform_color([0, 0, 0])
    pcd2 = pcd2.translate((0, 0.2, 0))

    modes = {'landmark': truth, 'points68': output}
    pcds = {'landmark': pcd2, 'points68': pcd1}
    for key, box in pin_boxes.items():
        print(key)
        for mode, _ in modes.items():
            box.switch_mode(mode)
            # for pin in box():
            colors = np.asarray(pcds[mode].colors)
            colors[box()] = (1, 0, 0)
            pcds[mode].colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcds[mode]])
            pcds[mode].paint_uniform_color([0, 0, 0])


def fit_output(model_output, flame_output, aligner, vis=False):
    y_pred, y_true = [], []
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
        y_pred.append(model_landmark)
        y_true.append(label_landmark)
    return y_pred, y_true


def FLAEP_loss(output_list, truth_list, pin_boxes):
    # draw_pinbox(output, truth, pin_boxes)
    shape = np.zeros((len(output_list), 7))
    epxression = np.zeros((len(output_list), 11))
    jaw = np.zeros((len(output_list), 3))

    pred = {'shape': shape, 'expression': epxression, 'jaw': jaw}
    label = copy.deepcopy(pred)
    mapping = {'points68': 'pred', 'landmark': 'truth'}
    losses = {'pred': pred, 'truth': label}
    for b, output, truth in zip(range(0, len(output_list)), output_list, truth_list):
        modes = {'landmark': truth, 'points68': output}
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

            # 코 중심으로부터의 눈가 거리
            dist = np.linalg.norm(eyes_points[0] - nose_tip)
            losses[mapping[mode]]['shape'][b, 0] = dist / base
            dist = np.linalg.norm(eyes_points[1] - nose_tip)
            losses[mapping[mode]]['shape'][b, 1] = dist / base
            dist = np.linalg.norm(eyes_points[4] - nose_tip)
            losses[mapping[mode]]['shape'][b, 2] = dist / base
            dist = np.linalg.norm(eyes_points[5] - nose_tip)
            losses[mapping[mode]]['shape'][b, 3] = dist / base

            # 코 중심으로부터의 입 양 끝 거리
            dist = np.linalg.norm(lip_points[0] - nose_tip)
            losses[mapping[mode]]['shape'][b, 4] = dist / base
            losses[mapping[mode]]['expression'][b, 0] = dist / base
            losses[mapping[mode]]['jaw'][b, 0] = dist / base
            dist = np.linalg.norm(lip_points[1] - nose_tip)
            losses[mapping[mode]]['shape'][b, 5] = dist / base
            losses[mapping[mode]]['expression'][b, 1] = dist / base
            losses[mapping[mode]]['jaw'][b, 1] = dist / base

            # 입의 벌린 정도, 입의 길이
            losses[mapping[mode]]['jaw'][b, 2] = lip_height / base
            dist = np.linalg.norm(lip_points[0] - lip_points[1])
            losses[mapping[mode]]['shape'][b, 6] = dist / base
            losses[mapping[mode]]['expression'][b, 10] = dist / base

            # 눈의 높이(2, 3), 길이(0, 1)
            dist = np.linalg.norm(eyes_points[1] - eyes_points[0])
            losses[mapping[mode]]['expression'][b, 2] = dist / base
            dist = np.linalg.norm(eyes_points[2] - eyes_points[3])
            losses[mapping[mode]]['expression'][b, 3] = dist / base
            dist = np.linalg.norm(eyes_points[5] - eyes_points[4])
            losses[mapping[mode]]['expression'][b, 4] = dist / base
            dist = np.linalg.norm(eyes_points[6] - eyes_points[7])
            losses[mapping[mode]]['expression'][b, 5] = dist / base

            # 코 중심으로부터의 눈썹의 거리
            dist = np.linalg.norm(eyebrow_points[1] - nose_tip)
            losses[mapping[mode]]['expression'][b, 6] = dist / base
            dist = np.linalg.norm(eyebrow_points[3] - nose_tip)
            losses[mapping[mode]]['expression'][b, 7] = dist / base
            dist = np.linalg.norm(eyebrow_points[5] - nose_tip)
            losses[mapping[mode]]['expression'][b, 8] = dist / base
            dist = np.linalg.norm(eyebrow_points[7] - nose_tip)
            losses[mapping[mode]]['expression'][b, 9] = dist / base

    for label, loss in losses.items():
        for loss_name, value in loss.items():
            if "pred" in label:
                losses[label][loss_name] = torch.tensor(value, requires_grad=True)
            else:
                losses[label][loss_name] = torch.tensor(value)
    return losses


def loop(batch_size=4, epochs=300, learning_rate=1e-3):
    # Initialize
    dataset_root = r'D:\Creadto\Heritage\Dataset\GAN dataset'

    with open(os.path.join(dataset_root, 'label.txt'), "r") as f:
        labels = f.readlines()
    labels = [label.replace('\n', '') for label in labels]
    with open("graph_info/edge_info.json", "r") as f:
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
    model = FLAEPv1()
    pre_loss = Aligner(pin_boxes=pin_boxes)
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # Dryrun model
    device = summary_device()
    model.to(device)
    generator.to(device)

    images, graphs, landmarks = next(iter(loader))
    output = model(images, graphs)
    vertices, landmark = generator(**output)
    # visualize_output(vertices=vertices, landmark=landmark, faces=generator.faces)
    y_pred, y_true = fit_output(landmarks, (vertices, landmark), pre_loss)
    losses = FLAEP_loss(y_pred, y_true, pin_boxes)
    shape_loss = loss(losses['pred']['shape'], losses['truth']['shape'])
    expression_loss = loss(losses['pred']['expression'], losses['truth']['expression'])
    jaw_loss = loss(losses['pred']['jaw'], losses['truth']['jaw'])
    loss_sum = shape_loss + expression_loss + jaw_loss
    print(loss_sum)

    best_accuracy = 0.0
    dataframe = pd.DataFrame(columns=['epoch', 'tick', 'shape', 'expression', 'jaw'])
    for tick in range(epochs):
        running_loss = {'shape': 0.0, 'expression': 0.0, "jaw": 0.0}
        pbar = tqdm(loader, desc='Epoch', position=0)
        for i, (images, graphs, landmarks) in enumerate(pbar):
            if len(images) != batch_size:
                continue
            optimizer.zero_grad()
            output = model(images, graphs)
            vertices, landmark = generator(**output)
            y_pred, y_true = fit_output(landmarks, (vertices, landmark), pre_loss)
            losses = FLAEP_loss(y_pred, y_true, pin_boxes)
            shape_loss = loss(losses['pred']['shape'], losses['truth']['shape'])
            expression_loss = loss(losses['pred']['expression'], losses['truth']['expression'])
            jaw_loss = loss(losses['pred']['jaw'], losses['truth']['jaw'])
            loss_sum = shape_loss + expression_loss + jaw_loss
            loss_sum.backward()
            optimizer.step()
            running_loss['epoch'] = tick
            running_loss['tick'] = i
            running_loss['shape'] = (running_loss['shape'] + shape_loss.item()) / 2.0
            running_loss['expression'] = (running_loss['expression'] + expression_loss.item()) / 2.0
            running_loss['jaw'] = (running_loss['jaw'] + jaw_loss.item()) / 2.0
            line = "Epoch: %03d, avg_loss(shape): %.4f, avg_loss(expression): %.4f, avg_loss(jaw): %.4f, ticks: %06d" % (tick, running_loss['shape'], running_loss['expression'], running_loss['jaw'], i)
            pbar.set_description(line)
            if i % 1000 == 0:
                series = pd.Series(running_loss)
                dataframe = pd.concat([dataframe.T, series], axis=1).T
                dataframe.to_csv('./train_log.csv')
            if i % 10000 == 0:
                now = datetime.now()
                now_str = now.strftime('%Y-%m-%d %H%M%S') + '.pth'
                torch.save(model.state_dict(), os.path.join(r'checkpoints', now_str))
        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d %H%M%S') + '.pth'
        torch.save(model.state_dict(), os.path.join(r'checkpoints', now_str))
    # evaluation
    # utils-display, lod, checkpoints
    pass


if __name__ == "__main__":
    loop()
