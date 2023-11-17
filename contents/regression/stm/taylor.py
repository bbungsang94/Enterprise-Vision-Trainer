import copy

import numpy as np
import open3d as o3d
import torch


class Taylor:
    def __init__(self, tape, pin, circ_dict, model_dict=None):
        """
        :param tape: It's interaction list from convention.py. call get_interactions
        :param pin: Vertices index about human parts in 3d mesh model load standing/sitting.json
        """
        self.tape = tape
        self.pin = self._convert_tag(pin)
        self.table = torch.zeros(1, len(self.tape))
        self.model = model_dict
        self.circ_dict = circ_dict

    def update(self, model_dict):
        """
        :param model_dict: vertices ("standing", "sitting", "t", "hands-on", "curve")
        :return:
        """
        self.model = model_dict
        self.table = torch.zeros(self.model['t'].shape[0], len(self.tape))

    def order(self, gender, fast=False, visualize=False):
        only_female = ["Waist Height natural indentation", "Underbust Circumference"]
        male_indexes = np.where(np.array(gender) == "male")[0]
        for i, paper in enumerate(self.tape):
            kor, eng, tags, func, pose = paper

            # points name to index
            args = []
            for tag in tags:
                index = self.pin[tag]
                point = self.model[pose][:, index]
                args.append(point)

            # function 인식
            if fast and ("circ" in func or "length" in func):
                continue
            if "circ" in func:
                stub = func.split('-')
                indexes = self.circ_dict[eng]
                args.clear()
                for index in indexes:
                    args.append(self.model[pose][:, index])
                args.insert(0, stub[-1])
                func = stub[0]

            value = getattr(self, func)(*args)
            self.table[:, i] = value

            if eng in only_female:
                self.table[male_indexes, i] = 0.

        if visualize:
            for pose, vertex in self.model.items():
                points = []
                lines = []
                line_colors = []
                count = 0
                v = vertex[0].numpy()
                colors = np.zeros((v.shape[0], 3))
                colors += 0.4
                for paper in self.tape:
                    _, eng, tags, func, p = paper
                    if p != pose:
                        continue
                    if "circ" in func:
                        indexes = self.circ_dict[eng]
                    else:
                        indexes = [self.pin[tag] for tag in tags]
                    for i, index in enumerate(indexes):
                        point = copy.deepcopy(v[index])
                        if "width" in func:
                            point[1] = v[indexes[0]][1]
                            point[2] = v[indexes[0]][2]
                            line_colors.append([1., 0., 0.])
                        elif "depth" in func:
                            point[0] = v[indexes[0]][0]
                            point[1] = v[indexes[0]][1]
                            line_colors.append([0., 1., 0.])
                        elif "height" in func:
                            point[0] = v[indexes[0]][0]
                            point[2] = v[indexes[0]][2]
                            line_colors.append([0., 0., 1.])
                        else:
                            line_colors.append([1., 0., 1.])
                        points.append(point)
                        lines.append([i + count, ((i + 1) % len(indexes)) + count])
                    count += len(indexes)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(v)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(np.array(points))
                line.lines = o3d.utility.Vector2iVector(np.array(lines))
                line.colors = o3d.utility.Vector3dVector(np.array(line_colors))
                o3d.visualization.draw_geometries([line, pcd])

        return self.table

    def _convert_tag(self, pin):
        standing, sitting = pin
        tag = dict()
        for key, value in standing.items():
            _, eng, direction = self._separate_key(key)
            tag[eng + direction] = value

        for key, value in sitting.items():
            _, eng, direction = self._separate_key(key)
            tag[eng + direction] = value
        return tag

    @staticmethod
    def _separate_key(name):
        stub = name.split(', ')
        if len(stub[-1]) > 1:
            direction = ""
        else:
            direction = ", " + stub[-1]
        return stub[0], stub[1], direction

    @staticmethod
    def width(a, b):
        return abs(a[:, 0] - b[:, 0])

    @staticmethod
    def height(a, b):
        return abs(a[:, 1] - b[:, 1])

    @staticmethod
    def depth(a, b):
        return abs(a[:, 2] - b[:, 2])

    @staticmethod
    def circ(direction, *args):
        pivot = {
            'v': 0,
            'h': 1,
        }
        if direction in pivot:
            fix = args[0][:, pivot[direction]]
        else:
            fix = np.NaN

        length = len(args)
        result = torch.zeros(args[0].shape[0])
        for i in range(length):
            a = args[i]
            b = args[(i + 1) % length]
            if direction in pivot:
                a[:, pivot[direction]] = fix
                b[:, pivot[direction]] = fix
            result += torch.linalg.vector_norm(b - a, dim=1)
            # 직선거리 계산
        return result

    @staticmethod
    def length(*args):
        length = len(args)
        result = torch.zeros(args[0].shape[0])
        for i in range(length):
            a = args[i]
            if (i + 1) == length:
                break
            b = args[i + 1]
            result += torch.linalg.vector_norm(b - a, dim=1)
        return result
