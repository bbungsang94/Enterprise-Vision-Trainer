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
                if visualize:
                    print(tag)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(self.model[pose][0].numpy())
                    colors = np.zeros((len(self.model[pose][0]), 3))
                    colors[index, 0] = 1.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    o3d.visualization.draw_geometries([pcd])

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
            
            if visualize:
                # 이거 반드시 봐야함
                pass

            self.table[:, i] = value

            if eng in only_female:
                self.table[male_indexes, i] = 0.

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
            result += torch.norm(b - a, dim=1)
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
            result += torch.norm(b - a, dim=1)
        return result
