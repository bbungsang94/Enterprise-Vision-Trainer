import os
import cv2
import time
from model import FaceLandMarks


def main(image_files):
    detector = FaceLandMarks()
    for image_file in image_files:
        image = cv2.imread(image_file)
        begin = time.time()
        img, faces = detector.findFaceLandmark(image)
        if len(faces) != 0:
            print(len(faces))

        cv2.putText(img, 'Inference time: %.3f' % (time.time() - begin),
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Landmarks", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    files = os.listdir('../images')
    files = [os.path.join('../images', x) for x in files]
    main(files)