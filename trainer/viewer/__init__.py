from functools import partial
from trainer.viewer.geometry import Open3dViewer, FLAEPViewer
from trainer.viewer.image import DiffusionViewer, ImageViewer
from trainer.viewer.base import Base


def get_viewer_fn(viewer, **kwargs) -> Base:
    return viewer(**kwargs)


REGISTRY = {'Open3D': partial(get_viewer_fn, viewer=Open3dViewer),
            'FLAEP': partial(get_viewer_fn, viewer=FLAEPViewer),
            'Diffusion': partial(get_viewer_fn, viewer=DiffusionViewer),
            'Image': partial(get_viewer_fn, viewer=ImageViewer)
            }
