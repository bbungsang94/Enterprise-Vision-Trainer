
class FakeLoss:
    def __init__(self):
        pass

    def __call__(self, loss, fake):
        return loss
