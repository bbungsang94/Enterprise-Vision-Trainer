from functools import partial
from trainer.viewer.geometry import Open3dViewer, FLAEPViewer
from trainer.viewer.image import DiffusionViewer, ImageViewer, LandmarkViewer
from trainer.viewer.base import Base, Concrete


def get_viewer_fn(viewer, **kwargs) -> Base:
    return viewer(**kwargs)


REGISTRY = {
    'None': partial(get_viewer_fn, viewer=Concrete),
    'Open3D': partial(get_viewer_fn, viewer=Open3dViewer),
    'FLAEP': partial(get_viewer_fn, viewer=FLAEPViewer),
    'Diffusion': partial(get_viewer_fn, viewer=DiffusionViewer),
    'Image': partial(get_viewer_fn, viewer=ImageViewer),
    'Landmark': partial(get_viewer_fn, viewer=LandmarkViewer)
            }
