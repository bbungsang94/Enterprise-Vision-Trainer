import os
import cv2
import open3d as o3d
import numpy as np
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
            o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    IMAGE_PATH = r'D:\Creadto\Utilities\Enterprise-Vision-Trainer\facial_landmarks\images'
    files = [os.path.join(IMAGE_PATH, x) for x in os.listdir(IMAGE_PATH)]
    main(files)
