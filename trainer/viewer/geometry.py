from typing import Optional, Type

from open3d.cpu.pybind.geometry import PointCloud

from trainer.viewer.base import Base
import open3d as o3d
import numpy as np


class Open3dViewer(Base):
    def __init__(self):
        super().__init__()
        self.platform = 'Open3D'

    def show(self, **kwargs):

        pass

    def save(self):
        pass

    def summary(self):
        pass

    @staticmethod
    def image_to_pcd(image: np.ndarray, scale=1.0, depth_offset=0) -> Type[PointCloud] | None:
        # image는 x, y, color
        # z는 offset
        channel, row_len, column_len = image.shape
        if channel != 3:
            print("Invalid shape")
            return None
        max_len = max(row_len, column_len)
        points = np.zeros((row_len * column_len, 3), dtype=float)
        colors = np.zeros((row_len * column_len, 3), dtype=float)
        counter = 0
        for row in range(row_len):
            for column in range(column_len):
                x = column / max_len * scale
                y = row / max_len * scale
                points[counter] = [x, y, depth_offset]
                colors[counter] = [image[0, row, counter], image[1, row, counter], image[2, row, counter]]
        pcd = o3d.geometry.PointCloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
