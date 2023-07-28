from trainer.viewer.base import Base


class Open3dViewer(Base):
    def __init__(self):
        super().__init__()
        self.platform = 'Open3D'

    def show(self, **kwargs):
        pass

    def save(self):
        pass

    def summary(self):
        pass
