import os
from typing import Tuple
from tqdm import tqdm
import cv2
import time
import numpy as np
import pandas as pd
from contents.facial_landmarks.utility import make_3d_points
from contents.facial_landmarks.cv_mesh.model import FaceLandMarks


def inference(model: FaceLandMarks, input_data: np.ndarray) -> Tuple[dict, float]:
    begin = time.time()
    img, faces, landmarks = model.findFaceLandmark(input_data)
    result = {'image': img, 'faces': faces, 'landmarks': landmarks}
    return result, (time.time() - begin)


def main():
    detector = FaceLandMarks()
    dataset_path = r"D:\Creadto\Heritage\Dataset\GAN dataset\image"
    debug_output = r"D:\Creadto\Heritage\Dataset\GAN dataset\debug-468"
    landmark_output = r"D:\Creadto\Heritage\Dataset\GAN dataset\landmark-468"
    image_files = os.listdir(dataset_path)
    for image_file in tqdm(image_files):
        image = cv2.imread(os.path.join(dataset_path, image_file))
        result, inference_time = inference(detector, image)
        cv2.imwrite(os.path.join(debug_output, image_file), result['image'])
        points = make_3d_points(result['landmarks'])
        landmark = pd.DataFrame(points[0])
        landmark.to_csv(os.path.join(landmark_output, image_file.replace('.jpg', '.csv')),
                        index=False, header=False)


if __name__ == "__main__":
    main()
