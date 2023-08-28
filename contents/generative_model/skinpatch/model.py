import torch
from contents import Base


class SkinDDPM(Base):
    def __init__(self, **kwargs):
        pass

    def __call__(self, x):
        pass

    def to(self, device: torch.device):
        pass

    def forward(self, x) -> [torch.tensor]:
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def parameters(self):
        pass

    def state_dict(self, **kwargs):
        pass

    def load_state_dict(self, weights):
        pass
