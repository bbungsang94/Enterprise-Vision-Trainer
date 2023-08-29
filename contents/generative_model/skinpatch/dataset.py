import os
import timm
from torch.utils.data.dataloader import T_co
import torch
from torch.utils.data import Dataset, DataLoader


class SkinDataset(Dataset):
    def __init__(self, dataset_root):
        self.root = dataset_root
        self.x_data = os.listdir(self.root)
        self.x_data = [os.path.join(self.root, x) for x in self.x_data]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        full_path = self.x_data[idx]
        values = torch.load(full_path)
        image = values['image']
        skin = values['skin']
        return image, skin


class SkinDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co],
                 n_steps=1000,
                 min_beta=10 ** -4, max_beta=0.02, colour=True,
                 **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.extractor = timm.create_model('resnet50d', pretrained=True, num_classes=0)
        # Utilities
        self.colour = 3 if colour is True else 1
        self.n_steps = n_steps

        # Number of steps is typically in the order of thousands
        self.betas = torch.linspace(min_beta, max_beta, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device
        self.extractor = self.extractor.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)

    def state_dict(self):
        result = {'resnet50d': self.extractor.state_dict()}

        return result

    def load_state_dict(self, weights):
        self.extractor.load_state_dict(weights['resnet50d'])

    def sample(self):
        indexes = torch.randint(0, len(self.dataset), (self.batch_size, 1))
        images = []
        skins = []
        for index in indexes:
            sample = self.dataset[index]
            image, skin = sample
            images.append(image)
            skins.append(skin)

        images = torch.stack(images, dim=0)
        skins = torch.stack(skins, dim=0)
        result = {'images': images,
                  'title': "Input image samples"
                  }
        return result

    def _collate_fn(self, batch) -> [torch.tensor, torch.tensor]:
        batch_images = []
        batch_skins = []

        for stub in batch:
            batch_images.append(stub[0])
            batch_skins.append(stub[1])

        batch_images = torch.stack(batch_images, dim=0).to(self.device)
        batch_skins = torch.stack(batch_skins, dim=0).to(self.device)
        t = torch.randint(0, self.n_steps, (len(batch),)).to(self.device)

        # processing image to get raw latent space as pure noise
        x0 = self.extractor(batch_images)
        multiple = 2 ** 16 * self.colour // x0.shape[1]
        x0 = x0.unsqueeze(2).expand(-1, -1, multiple) # [b, c * 256 * 256]
        x0 = x0.reshape(len(batch), 3, 256, 256)

        # make noise(diffusion forward)
        n, c, h, w = x0.shape
        epsilon = torch.randn(n, c, h, w).to(self.device)
        a_bar = self.alpha_bars[t]
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * epsilon

        return (noisy, t), batch_skins
