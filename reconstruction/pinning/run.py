import copy
import os
import cv2
import open3d as o3d
import numpy as np
from pins.pin import PinLoader
from utility.open3d_utils import draw_lines
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
        points = make_3d_points(landmark_result['landmarks'])
        for face_point in points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(face_point)
            pcd.paint_uniform_color([0, 0, 0])

            # Align point 추가
            align_box = pin_boxes['Align']
            index_align = align_box()
            lines = draw_lines(face_point[index_align])
            color = [1, 0, 1]
            colors = np.array([color] * len(index_align), dtype=np.float64)
            pcd_colors = np.asarray(pcd.colors)
            pcd_colors[index_align] = colors
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)  # Ensure pcd.colors is not None
            o3d.visualization.draw_geometries([pcd, lines, coord])

            # Align test y축
            aligned = copy.deepcopy(pcd)
            left_face = align_box[2]
            right_face = align_box[3]
            fake_point = copy.deepcopy(face_point[left_face(), [0, 2]])
            fake_point[1] = face_point[right_face(), 2]
            right_point = copy.deepcopy(face_point[right_face(), [0, 2]])
            left_point = copy.deepcopy(face_point[left_face(), [0, 2]])
            if left_point[1] > right_point[1]:
                pos = -1
            else:
                pos = 1
            r_x = np.linalg.norm(right_point - fake_point, ord=2)
            r_y = np.linalg.norm(fake_point - left_point, ord=2)
            theta = np.arctan(r_y / r_x)
            r = aligned.get_rotation_matrix_from_xyz((0, pos * theta, 0))
            aligned.rotate(r, center=tuple(face_point[right_face()]))

            # update face points
            face_point = np.asarray(aligned.points)
            lines = draw_lines(face_point[index_align])
            o3d.visualization.draw_geometries([aligned, lines, coord])

            # Align test x축
            head = align_box[0]
            jaw = align_box[1]
            fake_point = copy.deepcopy(face_point[head(), [1, 2]])
            fake_point[1] = face_point[jaw(), 2]
            head_point = copy.deepcopy(face_point[head(), [1, 2]])
            jaw_point = copy.deepcopy(face_point[jaw(), [1, 2]])
            if jaw_point[1] > head_point[1]:
                pos = -1
            else:
                pos = 1
            r_x = np.linalg.norm(jaw_point - fake_point, ord=2)
            r_y = np.linalg.norm(fake_point - head_point, ord=2)
            theta = np.arctan(r_y / r_x)
            r = aligned.get_rotation_matrix_from_xyz((pos * theta, 0, 0))
            aligned.rotate(r, center=tuple(face_point[jaw()]))

            # update face points
            face_point = np.asarray(aligned.points)
            lines = draw_lines(face_point[index_align])
            o3d.visualization.draw_geometries([aligned, lines, coord])

            # Align test z축
            if head_point[0] < jaw_point[0]:
                r = aligned.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
                aligned.rotate(r)
                face_point = np.asarray(aligned.points)
                lines = draw_lines(face_point[index_align])
                o3d.visualization.draw_geometries([aligned, lines, coord])

            fake_point = copy.deepcopy(face_point[left_face(), [0, 1]])
            fake_point[1] = face_point[right_face(), 1]
            right_point = copy.deepcopy(face_point[right_face(), [0, 1]])
            left_point = copy.deepcopy(face_point[left_face(), [0, 1]])
            if right_point[1] > left_point[1]:
                pos = -1
            else:
                pos = 1
            r_x = np.linalg.norm(right_point - fake_point, ord=2)
            r_y = np.linalg.norm(fake_point - left_point, ord=2)
            theta = np.arctan(r_y / r_x)
            r = aligned.get_rotation_matrix_from_xyz((0, 0, pos * theta))
            aligned.rotate(r, center=tuple(face_point[right_face()]))

            # update face points
            face_point = np.asarray(aligned.points)
            lines = draw_lines(face_point[index_align])
            o3d.visualization.draw_geometries([aligned, lines, coord])

            # 마지막 offset 처리
            nose_tip = align_box[-1]
            center_point = copy.deepcopy(face_point[nose_tip()])
            min_bound = aligned.get_min_bound()
            center_point[2] = min_bound[2]
            face_point -= center_point
            aligned.points = o3d.utility.Vector3dVector(face_point)
            o3d.visualization.draw_geometries([aligned, coord])

            scene = o3d.io.read_triangle_model('./base heads/high poly head.obj')
            mesh = scene.meshes[0].mesh
            o3d.visualization.draw_geometries([mesh, coord])


if __name__ == "__main__":
    IMAGE_PATH = r'D:\Creadto\Utilities\Enterprise-Vision-Trainer\facial_landmarks\images'
    files = [os.path.join(IMAGE_PATH, x) for x in os.listdir(IMAGE_PATH)]
    main(files)
