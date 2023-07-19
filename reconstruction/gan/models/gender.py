import torch
import torch.nn as nn
import argparse
from torchvision import models
from classification.mivolo.predictor import Predictor


class Resnet18Classifier(nn.Module):
    def __init__(self, load_path=None):
        super(Resnet18Classifier, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device object
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        self.model.to(self.device)

    def forward(self, x):
        y = self.model(x)
        output = torch.argmax(y).item()
        gender = "female" if output == 1 else "male"
        return gender


class MiVOLOClassifier:
    def __init__(self, **kwargs):
        args = self.get_parser(kwargs['detector_weights'], kwargs['checkpoint'], kwargs['device_name'])
        self.predictor = Predictor(args, verbose=True)

    def __call__(self, x):
        return self.predictor.recognize(x)

    @staticmethod
    def get_parser(detector_weights, checkpoint, device_name):
        parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
        parser.add_argument("--detector-weights", type=str, default=detector_weights, help="Detector weights (YOLOv8).")
        parser.add_argument("--checkpoint", default=checkpoint, type=str, help="path to mivolo checkpoint")
        parser.add_argument(
            "--with-persons", action="store_true", default=True,
            help="If set model will run with persons, if available"
        )
        parser.add_argument(
            "--disable-faces", action="store_true", default=False,
            help="If set model will use only persons if available"
        )
        parser.add_argument("--device", default=device_name, type=str, help="Device (accelerator) to use.")

        return parser.parse_args()
