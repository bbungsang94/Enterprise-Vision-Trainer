import os
import cv2
import open3d as o3d
import numpy as np
import pandas as pd
from pins.pin import PinLoader
from facial_landmarks.cv_mesh.run import inference as landmark_model
from facial_landmarks.cv_mesh.model import FaceLandMarks


def make_3d_points(landmarks) -> list:
    batch = []
    for faceLms in landmarks:
        face = np.zeros((len(faceLms.landmark), 3))
        for index, lm in enumerate(faceLms.landmark):
            face[index, :] = [lm.x, lm.y, lm.z]
        batch.append(face)
    return batch


def main(image_files):
    pin_boxes = PinLoader.load_pins(path='./pins', filename='pin_info.json')
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    for filename in image_files:
        image = cv2.imread(filename)
        landmark_result, inference_time = landmark_model(FaceLandMarks(), image)
        cv2.putText(landmark_result['image'], 'Inference time: %.3f' % inference_time,
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Landmarks", landmark_result['image'])
        cv2.waitKey(1)

        points = make_3d_points(landmark_result['landmarks'])
        for face_point in points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(face_point)
            dataframe = pd.DataFrame(face_point)
            dataframe.to_csv('./landamrks.csv')

            pcd.paint_uniform_color([0, 0, 0])

            indices_eye = [133, 33, 159, 145, 382, 263, 386, 374]
            indices_nose = [1, 129, 358]
            indices_lip = [61, 291, 13, 14, 0, 16]
            indices_eyebrow = [107, 55, 70, 46, 336, 285, 300, 276]

            indices_to_color = indices_eye + indices_nose + indices_lip + indices_eyebrow
            color = [1, 0, 0]
            pcd_colors = np.asarray(pcd.colors)
            colors = np.array([color] * len(indices_to_color), dtype=np.float64)
            pcd_colors[indices_to_color] = colors

            pcd.paint_uniform_color([0, 0, 0])
            index_align = [10, 1]
            color = [1, 0, 1]
            colors = np.array([color] * len(index_align), dtype=np.float64)
            pcd_colors = np.asarray(pcd.colors)
            pcd_colors[index_align] = colors
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)  # Ensure pcd.colors is not None


            o3d.visualization.draw_geometries([pcd, coord])
            o3d.io.write_point_cloud('./base heads/inference.ply', pcd)

            scene = o3d.io.read_triangle_model('./base heads/high poly head.obj')
            mesh = scene.meshes[0].mesh
            o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    IMAGE_PATH = r'D:\Creadto\Utilities\Enterprise-Vision-Trainer\facial_landmarks\images'
    files = [os.path.join(IMAGE_PATH, x) for x in os.listdir(IMAGE_PATH)]
    main(files)
