import os
from typing import Tuple

import cv2
import time
import numpy as np
from facial_landmarks.cv_mesh.model import FaceLandMarks


def inference(model: FaceLandMarks, input_data: np.ndarray) -> Tuple[dict, float]:
    begin = time.time()
    img, faces, landmarks = model.findFaceLandmark(input_data)
    result = {'image': img, 'faces': faces, 'landmarks': landmarks}
    return result, (time.time() - begin)


def main(image_files):
    detector = FaceLandMarks()
    for image_file in image_files:
        image = cv2.imread(image_file)
        result, inference_time = inference(detector, image)
        if len(result['faces']) != 0:
            print(len(result['faces']))

        cv2.putText(result['image'], 'Inference time: %.3f' % inference_time,
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Landmarks", result['image'])
        cv2.waitKey(1)


if __name__ == "__main__":
    files = os.listdir('../images')
    files = [os.path.join('../images', x) for x in files]
    main(files)
