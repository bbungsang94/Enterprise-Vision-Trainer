import os
from tqdm import tqdm
import open3d as o3d


def main():
    root = r"D:\Creadto\Utilities\Enterprise-Vision-Trainer\dummy\upsampler"
    obj_files = os.listdir(root)
    for obj_file in tqdm(obj_files):
        scene = o3d.io.read_triangle_model(os.path.join(root, obj_file))
        mesh = scene.meshes[0].mesh
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        # o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(os.path.join(root, 'Up-'+obj_file), mesh)


if __name__ == "__main__":
    main()
