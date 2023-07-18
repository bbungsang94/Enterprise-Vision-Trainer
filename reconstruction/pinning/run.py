import os
import cv2
import numpy as np
from model.align import Aligner
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
    for filename in image_files:
        image = cv2.imread(filename)
        landmark_result, inference_time = landmark_model(FaceLandMarks(), image)
        points = make_3d_points(landmark_result['landmarks'])
        for face_point in points:
            aligner = Aligner(pin_boxes=pin_boxes, points=face_point)
            pcd = aligner.run_seq()


if __name__ == "__main__":
    IMAGE_PATH = r'D:\Creadto\Utilities\Enterprise-Vision-Trainer\facial_landmarks\images'
    files = [os.path.join(IMAGE_PATH, x) for x in os.listdir(IMAGE_PATH)]
    main(files)
