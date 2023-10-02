import numpy as np
import open3d as o3d
import torch

from contents.reconstruction.fleap.model import FLAMESet


def main():
    batch_size = 2
    genders = ['male', 'male']
    parameter = {
        "root": "D:\\Creadto\\Utilities\\Enterprise-Vision-Trainer\\contents\\reconstruction\\gan\\models\\pretrained",
        "flame_model_path": "generic_model.pkl",
        "static_landmark_embedding_path": "flame_static_embedding.pkl",
        "dynamic_landmark_embedding_path": "flame_dynamic_embedding.npy",
        "shape_params": 300,
        "expression_params": 100,
        "pose_params": 3,
        "use_face_contour": True,
        "use_3D_translation": True,
        "optimize_eyeballpose": True,
        "optimize_neckpose": True,
        "num_worker": 4,
        "ring_margin": 0.5,
        "ring_loss_weight": 1.0,
        "batch_size": batch_size
    }

    generator = FLAMESet(**parameter)
    for _ in range(10):
        shape = torch.ones(1, parameter['shape_params'])
        shape = torch.concat([shape, torch.ones(1, parameter['shape_params'])], dim=0)
        shape *= 4
        shape -= 2
        expression = torch.zeros(batch_size, parameter['expression_params'])
        output = {'batch_size': batch_size, 'shape_params': shape, 'expression_params': expression}
        vertices, landmarks = generator(genders=genders, **output)
        face = generator.faces

        tip = landmarks[0, 13, :]
        left_vertices = (vertices[0] - tip).detach()
        tip = landmarks[1, 13, :]
        right_vertices = (vertices[1] - tip).detach()

        # make 2 faces
        points = [[-0.002, -1.0, 0], [-0.002, 1.0, 0], [-0.002, -1.0, 0.0], [-0.002, 1.0, -1.0]]
        faces = [[0, 1, 2], [1, 2, 3]]
        surface = o3d.geometry.TriangleMesh()
        surface.vertices = o3d.utility.Vector3dVector(points)
        surface.triangles = o3d.utility.Vector3iVector(faces)
        surface.paint_uniform_color([1, 0, 0])

        right = o3d.geometry.PointCloud()
        right.points = o3d.utility.Vector3dVector(left_vertices.detach())
        right.paint_uniform_color([0, 0, 0])

        check = torch.where(abs(right_vertices[:, 0]) < 0.002)
        selected = check[0]
        print(len(selected))
        right_colour = np.asarray(right.colors)
        right_colour[selected] = [1, 0, 0]
        right.colors = o3d.utility.Vector3dVector(right_colour)

        o3d.visualization.draw_geometries([right])

        # merge 2 faces
        check = torch.where(right_vertices[:, 0] > 0)
        selected = check[0]
        print(len(selected))
        left_vertices[selected, :] = right_vertices[selected, :].detach()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(left_vertices.detach())
        mesh.triangles = o3d.utility.Vector3iVector(face)

        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(vertices[0].detach())
        # mesh.triangles = o3d.utility.Vector3iVector(face)
        #
        # side = o3d.geometry.TriangleMesh()
        # side.vertices = o3d.utility.Vector3dVector(vertices[1].detach() + 0.2)
        # side.triangles = o3d.utility.Vector3iVector(face)

        coordi = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([mesh, coordi])


if __name__ == '__main__':
    main()
