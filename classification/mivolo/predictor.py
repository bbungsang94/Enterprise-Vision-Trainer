from typing import Optional, Tuple

import numpy as np
from .model.mi_volo import MiVOLO
from .model.yolo_detector import Detector, PersonAndFaceResult


class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )

    def recognize(self, image: np.ndarray) -> PersonAndFaceResult:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        return detected_objects
