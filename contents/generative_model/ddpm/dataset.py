from functools import partial
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from torch import nn
from torch.utils.data import Dataset

from modules.utils import convert_image_to_fn, exists


class DDPMDataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            augment_horizontal_flip=False,
            convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
