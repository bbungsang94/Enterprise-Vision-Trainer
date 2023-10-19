
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import T_co


class GenLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], **kwargs):
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    def _collate_fn(self, batch) -> [torch.tensor, torch.tensor]:
        batch_images = self.make_batch(batch)
        return batch_images.detach(), batch_images.detach()

    def make_batch(self, batch):
        batch_images = []

        for stub in batch:
            trans = torchvision.transforms.ToTensor()
            image = trans(stub[0])
            batch_images.append(image)

        batch_images = torch.stack(batch_images, dim=0).to(self.device)
        return batch_images

    def sample(self):
        indexes = torch.randint(0, len(self.dataset), (self.batch_size, 1))
        batch = []
        for index in indexes:
            batch.append(self.dataset[index])

        batch_images = self.make_batch(batch)

        result = {'images': batch_images,
                  'title': "Input image samples",
                  'device': self.device
                  }
        return result


class Gen4xLoader(GenLoader):
    def _collate_fn(self, batch) -> [torch.tensor, torch.tensor]:
        images, labels = self.make_batch(batch)
        return images.detach(), labels.detach()

    def make_batch(self, batch):
        batch_images = []
        batch_labels = []

        for stub in batch:
            trans = torchvision.transforms.ToTensor()
            width, height = stub[0].size
            target = (width * 4, height * 4)
            label = stub[0].resize(target)
            label = trans(label)
            image = trans(stub[0])
            batch_images.append(image)
            batch_labels.append(label)

        batch_images = torch.stack(batch_images, dim=0).to(self.device)
        batch_labels = torch.stack(batch_labels, dim=0).to(self.device)
        return batch_images, batch_labels

    def sample(self):
        pass