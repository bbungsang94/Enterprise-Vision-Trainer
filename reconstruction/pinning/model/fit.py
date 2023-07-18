import open3d as o3d


class Reconstructor:
    def __init__(self, basehead):
        head_mesh = o3d.io.read_triangle_model(basehead)
        head = head_mesh.meshes[0].mesh
        abb = head.get_axis_aligned_bounding_box()
        abb.color = (1, 0, 0)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(4)
        o3d.visualization.draw_geometries([abb, head, frame])
        test = 1


if __name__ == "__main__":
    recon = Reconstructor(basehead=r'D:\Creadto\Utilities\Enterprise-Vision-Trainer\reconstruction\pinning\base heads\high poly head.obj')

