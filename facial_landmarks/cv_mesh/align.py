import copy

from utility.open3d_utils import draw_lines
import numpy as np
import open3d as o3d


class Aligner:
    def __init__(self, pin_boxes, points=None):
        self._pin_boxes = pin_boxes
        pcd = o3d.geometry.PointCloud()
        self._pcd = pcd
        if points:
            self.update_points(points)

    def update_points(self, points):
        self._pcd.points = o3d.utility.Vector3dVector(points)
        self._pcd.paint_uniform_color([0, 0, 0])

    def align_seq(self):
        align_box = self._pin_boxes['Align']
        # y 얼라인
        self._pcd = self.align_xyz(pins=[align_box[2], align_box[3]], pivot=1)
        # x 얼라인
        self._pcd = self.align_xyz(pins=[align_box[1], align_box[0]], pivot=0)
        # z 얼라인
        self._pcd = self.align_xyz(pins=[align_box[2], align_box[3]], pivot=2)
        return self.get_points()

    def flip_seq(self):
        align_box = self._pin_boxes['Align']
        points = self.get_points()
        head_point = copy.deepcopy(points[align_box[0](), [1, 2]])
        jaw_point = copy.deepcopy(points[align_box[1](), [1, 2]])
        # flip
        if head_point[0] < jaw_point[0]:
            self._pcd = self.flip_yz()
        return self.get_points()

    def offset_seq(self):
        align_box = self._pin_boxes['Align']
        points = self.get_points()
        nose_tip = align_box[-1]
        center_point = copy.deepcopy(points[nose_tip()])
        min_bound = self._pcd.get_min_bound()
        center_point[2] = min_bound[2]
        points -= center_point
        self._pcd.points = o3d.utility.Vector3dVector(points)
        return self.get_points()

    def run_seq(self, show=False):
        self.align_seq()
        self.flip_seq()
        self.offset_seq()
        if show:
            self.show_pcds(True, self._pcd)
        return self.get_points()

    def get_points(self):
        return np.asarray(self._pcd.points)

    def show_align(self):
        align_box = self._pin_boxes['Align']
        index_align = align_box()
        lines = draw_lines(np.asarray(self._pcd.points)[index_align])
        color = [1, 0, 1]
        colors = np.array([color] * len(index_align), dtype=np.float64)
        pcd_colors = np.asarray(self._pcd.colors)
        pcd_colors[index_align] = colors
        self._pcd.colors = o3d.utility.Vector3dVector(pcd_colors)  # Ensure pcd.colors is not None
        self.show_pcds(True, self._pcd, lines)

    def flip_yz(self):
        # Align test z축
        aligned = copy.deepcopy(self._pcd)
        r = aligned.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
        aligned.rotate(r)
        return aligned

    def align_xyz(self, pins, pivot):
        alternatives = [0, 1, 2]
        alternatives.remove(pivot)
        points = np.asarray(self._pcd.points)

        alpha = pins[0]
        beta = pins[1]
        fake_point = copy.deepcopy(points[alpha(), alternatives])
        fake_point[1] = points[beta(), alternatives[-1]]
        alpha_point = copy.deepcopy(points[alpha(), alternatives])
        beta_point = copy.deepcopy(points[beta(), alternatives])
        if alpha_point[1] > beta_point[1]:
            direction = -1
        else:
            direction = 1
        r_x = np.linalg.norm(beta_point - fake_point, ord=2)
        r_y = np.linalg.norm(fake_point - alpha_point, ord=2)
        theta = np.arctan(r_y / r_x)

        aligned = copy.deepcopy(self._pcd)
        rot_xyz = np.array([0, 0, 0], dtype=np.float)
        rot_xyz[pivot] = 1
        rot_xyz *= (direction * theta)
        r = aligned.get_rotation_matrix_from_xyz(tuple(rot_xyz))
        aligned.rotate(r, center=tuple(points[beta()]))
        return aligned

    @staticmethod
    def show_pcds(coord: bool, *args):
        args = list(args)
        if coord:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            args.append(frame)
        o3d.visualization.draw_geometries(args)
