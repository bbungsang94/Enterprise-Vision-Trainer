import copy
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from contents.regression.stm.convention import get_interactions
from contents.regression.stm.pose import get_t, get_sitdown, get_curve, get_hands_on, get_standing
from contents.regression.stm.taylor import Taylor
from contents.regression.stm.utilities import rodrigues, with_zeros, pack


def note():
    #GNN으로 갈 필요가 있다.
    #다만 Regression model 일부를 수정해야함
    #ReLU를 사용하다보니 Batch norm을 ReLU뒤로 옮겨봤다
    pass


class STMRegression(nn.Module):
    def __init__(self, input_dim, output_dim, smpl_root, pin_root, circ_root):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 첫 번째 fully connected layer
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, 256)  # 두 번째 fully connected layer
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(256, 512)  # 세 번째 fully connected layer
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.fc4 = nn.Linear(512, output_dim)  # 출력 레이어

        self.relu = nn.ReLU()  # 활성화 함수로 ReLU 사용
        self.dropout = nn.Dropout(0.2)  # 드롭아웃 레이어

        male_path = os.path.join(smpl_root, "SMPLX_MALE.pkl")
        female_path = os.path.join(smpl_root, "SMPLX_FEMALE.pkl")
        male = SMPL(path=male_path)
        female = SMPL(path=female_path)
        self.body = {'male': male, 'female': female}

        with open(os.path.join(pin_root, 'standing.json'), 'r', encoding='UTF-8-sig') as f:
            standing = json.load(f)
        with open(os.path.join(pin_root, 'sitting.json'), 'r', encoding='UTF-8-sig') as f:
            sitting = json.load(f)
        with open(os.path.join(circ_root, 'circumference.json'), 'r', encoding='UTF-8-sig') as f:
            circ_dict = json.load(f)
        self.taylor = Taylor(tape=get_interactions(), pin=(standing, sitting), circ_dict=circ_dict)

    def forward(self, x, gender):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        x = self.bn3(x)
        # x = self.dropout(x)

        result = self.fc4(x)
        if self.training is False:
            result = self.do_taylor(shape=result, gender=gender)
            result = result.to(x.device)
        return None, result

    def do_taylor(self, shape, gender):
        batch_size = len(gender)
        shape = shape.cpu().to(torch.float64)
        poses = self._get_poses(batch_size=batch_size)
        offset = torch.zeros(batch_size, 3)

        bodies = dict()
        female_indexes = np.where(np.array(gender) == "female")[0]
        male_indexes = np.where(np.array(gender) == "male")[0]
        for key, value in poses.items():
            male_v, _ = self.body['male'](beta=shape,
                                          pose=value,
                                          offset=offset)
            female_v, _ = self.body['female'](beta=shape,
                                              pose=value,
                                              offset=offset)
            v = torch.zeros_like(male_v)
            v[female_indexes] = female_v[female_indexes]
            v[male_indexes] = male_v[male_indexes]

            bodies[key] = copy.deepcopy(v)
        self.taylor.update(model_dict=bodies)
        measure = self.taylor.order(gender=gender, visualize=False)
        table = measure / measure.max(dim=1).values.unsqueeze(1)
        return table

    @staticmethod
    def _get_poses(batch_size):
        result = {
            "t": get_t(batch_size),
            "standing": get_standing(batch_size),
            "sitting": get_sitdown(batch_size),
            "curve": get_curve(batch_size),
            "hands-on": get_hands_on(batch_size),
        }
        return result


class SMPL:
    def __init__(self, path):
        super(SMPL, self).__init__()
        with open(path, 'rb') as f:
            source = pickle.load(f, encoding="latin1")
        self.joint_regressor = torch.from_numpy(source['J_regressor']).type(torch.float64)
        self.weights = torch.from_numpy(source['weights']).type(torch.float64)
        self.posedirs = torch.from_numpy(source['posedirs']).type(torch.float64)
        self.v_template = torch.from_numpy(source['v_template']).type(torch.float64)
        self.shapedirs = torch.from_numpy(source['shapedirs']).type(torch.float64)
        self.kintree_table = source['kintree_table']
        self.faces = source['f']
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device

    def __call__(self, beta, pose, offset):
        batch_num = beta.shape[0]
        id_to_col = {self.kintree_table[1, i]: i
                     for i in range(self.kintree_table.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        v_shaped = torch.tensordot(beta, self.shapedirs, dims=([1], [2])) + self.v_template
        joint = torch.matmul(self.joint_regressor, v_shaped)
        r_cube_big = rodrigues(pose.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

        r_cube = r_cube_big[:, 1:, :, :]
        i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + torch.zeros((batch_num, r_cube.shape[1], 3, 3),
                                                                                   dtype=torch.float64)).to(self.device)
        lrotmin = (r_cube - i_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

        results = []
        results.append(with_zeros(torch.cat((r_cube_big[:, 0], torch.reshape(joint[:, 0, :], (-1, 3, 1))), dim=2)))
        for i in range(1, self.kintree_table.shape[1]):
            results.append(torch.matmul(results[parent[i]], with_zeros(
                torch.cat((r_cube_big[:, i], torch.reshape(joint[:, i, :] - joint[:, parent[i], :], (-1, 3, 1))),
                          dim=2))))

        stacked = torch.stack(results, dim=1)
        results = stacked - pack(torch.matmul(stacked, torch.reshape(
            torch.cat((joint, torch.zeros((batch_num, 55, 1), dtype=torch.float64).to(self.device)), dim=2),
            (batch_num, 55, 4, 1))))
        # Restart from here
        T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=torch.float64).to(self.device)), dim=2
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
        result = v + torch.reshape(offset, (batch_num, 1, 3))
        # estimate 3D joint locations
        # print(result.shape)
        # print(self.joint_regressor.shape)
        joints = torch.tensordot(result, self.joint_regressor, dims=([1], [1])).transpose(1, 2)
        return result, joints
