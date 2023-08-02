import os
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
        if kwargs:
            images = kwargs['image']
            vertices = kwargs['vertices']
            faces = kwargs['faces']
            text = kwargs['text']

            for i in range(len(images)):
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                background = self.convert_to_pcd(images[i])
                # vis.add_geometry(mesh)
                # vis.add_geometry(background)
                # vis.update_geometry()
                # vis.poll_events()
                # vis.update_renderer()
                o3d.visualization.draw_geometries([mesh, background])
                # if kwargs['save']:
                # vis.capture_screen_image(kwargs['save_path'])

                # vis.destroy_window()
                # vis.close()

    def save(self, **kwargs):
        pass

    def summary(self, **kwargs):
        pass

    @staticmethod
    def convert_to_pcd(image: np.ndarray, scale=0.5, depth_offset=-2.0) -> PointCloud | None:
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
                x = (column / max_len) * scale
                y = (row / max_len) * scale
                points[counter] = [x, y, depth_offset]
                colors[counter] = [image[0, row, column], image[1, row, column], image[2, row, column]]
                counter += 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd


class FLAEPViewer(Open3dViewer):
    def show(self, **kwargs):
        pass

    def save(self, **kwargs):
        if kwargs:
            images = kwargs['image']
            vertices = kwargs['vertices'].cpu().detach().numpy()
            faces = kwargs['faces']
            text = kwargs['text']

            for i in range(len(images)):
                vis = o3d.visualization.Visualizer()
                vis.create_window()

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                background = self.convert_to_pcd(images[i])
                vis.add_geometry(mesh)
                vis.add_geometry(background)
                vis.capture_screen_image(os.path.join(kwargs['save_path'], "%.3f-%02d" % (text, i) + '-result.png'))
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -3])
                ctr.set_up([0, 0, 0])
                ctr.set_zoom(1.5)
                vis.destroy_window()

    def summary(self, **kwargs):
        pass
