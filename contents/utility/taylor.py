import copy

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data, Batch


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
        self.table = torch.zeros(self.model[list(model_dict.keys())[-1]].shape[0], len(self.tape))

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
            if "circ" in func or 'length' in func:
                indexes = self.circ_dict[eng]
                args.clear()
                for index in indexes:
                    args.append(self.model[pose][:, index])

                if '-' in func:
                    stub = func.split('-')
                    args.insert(0, stub[-1])
                else:
                    stub = [func]
                    args.insert(0, 'straight')
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
                    elif "length" in func and eng in self.circ_dict:
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
        tag = dict()
        for dictionary in pin:
            for key, value in dictionary.items():
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

    def circ(self, direction, *args):
        result = self.length(direction, *args)

        pivot = {
            'v': 0,
            'h': 1,
        }
        if direction in pivot:
            fix = args[0][:, pivot[direction]]
        else:
            fix = None

        a = args[-1]
        b = args[0]
        if direction in pivot:
            if fix is not None:
                a[:, pivot[direction]] = fix
                b[:, pivot[direction]] = fix
        result += torch.linalg.vector_norm(b - a, dim=1)
        return result

    @staticmethod
    def length(direction, *args):
        pivot = {
            'v': 0,
            'h': 1,
        }
        if direction in pivot:
            fix = args[0][:, pivot[direction]]
        else:
            fix = None

        length = len(args)
        result = torch.zeros(args[0].shape[0])
        for i in range(length):
            a = args[i]
            if (i + 1) == length:
                break
            b = args[i + 1]
            if direction in pivot:
                if fix is not None:
                    a[:, pivot[direction]] = fix
                    b[:, pivot[direction]] = fix
            result += torch.linalg.vector_norm(b - a, dim=1)
        return result


class GraphTaylor(Taylor):
    def __init__(self, tape, pin, circ_dict):
        super().__init__(tape, pin, circ_dict)
        named_nodes = {}
        for i, key in enumerate(self.pin.keys()):
            named_nodes[key] = i
        edge_indexes = []
        for i, paper in enumerate(self.tape):
            kor, eng, tags, func, pose = paper
            if len(tags) > 2:
                raise "here!"
            edge_indexes.append([named_nodes[tags[0]], named_nodes[tags[1]]])
            edge_indexes.append([named_nodes[tags[1]], named_nodes[tags[0]]])

        self.named_nodes = named_nodes
        self.edge_indexes = torch.tensor(edge_indexes, dtype=torch.long).t()

        self.measure_code = {'width': -2, 'height': -1, 'depth': 0, 'length': 1, 'circ': 2,
                             'length-h': 1, 'length-v': 1, 'circ-h': 2, 'circ-v': 2}
        self.axis_code = {'width': -1, 'height': 0, 'depth': 1, 'length': 2, 'circ': 2,
                          'length-h': -1, 'length-v': 0, 'circ-h': -1, 'circ-v': 0}

    def get_measured_graph(self, gender, fast=False, visualize=False):
        batch_size = len(gender)
        measure = self.order(gender=gender, fast=fast, visualize=visualize)
        named_poses = {}
        for i, key in enumerate(self.model.keys()):
            named_poses[key] = i

        graphs = []
        for i in range(batch_size):
            edge_features = torch.zeros(len(self.tape) * 2, len(self.model) + 2)
            node_features = torch.zeros(len(self.pin), len(self.model) * 3)
            # node
            for key, vertices in self.model.items():
                v = vertices[i]
                for node_name, index in self.pin.items():
                    x, y, z = v[index]
                    pivot = named_poses[key] * 3
                    node_features[self.named_nodes[node_name], pivot] = x
                    node_features[self.named_nodes[node_name], pivot + 1] = y
                    node_features[self.named_nodes[node_name], pivot + 2] = z

                # edge feature
                for pivot, paper in enumerate(self.tape):
                    kor, eng, tags, func, pose = paper
                    edge_features[pivot * 2, named_poses[key]] = measure[i, pivot]
                    edge_features[pivot * 2, -1] = self.axis_code[func]
                    edge_features[pivot * 2, -2] = self.measure_code[func]

                    edge_features[pivot * 2 + 1, named_poses[key]] = measure[i, pivot]
                    edge_features[pivot * 2 + 1, -1] = self.axis_code[func]
                    edge_features[pivot * 2 + 1, -2] = self.measure_code[func]
            graph = Data(x=node_features,
                         edge_index=self.edge_indexes.contiguous(),
                         edge_attr=edge_features)
            if visualize:
                import networkx as nx
                import matplotlib.pyplot as plt

                # NetworkX 그래프로 변환
                G = nx.Graph()
                G.add_nodes_from(range(len(self.named_nodes)))
                G.add_edges_from(self.edge_indexes.t().tolist())

                # 노드 및 엣지 특성 추가
                for k in range(len(self.named_nodes)):
                    G.nodes[k]['features'] = node_features[k].numpy()

                for k, (src, tgt) in enumerate(self.edge_indexes.t().tolist()):
                    G[src][tgt]['features'] = edge_features[k].numpy()

                # 그래프 시각화
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=200, node_color='skyblue',
                        font_color='black', font_size=8)

                # 노드 및 엣지 특성 표시
                node_labels = nx.get_node_attributes(G, 'features')
                edge_labels = nx.get_edge_attributes(G, 'features')
                nx.draw_networkx_labels(G, pos, labels=node_labels)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                plt.show()
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        return measure, batch
