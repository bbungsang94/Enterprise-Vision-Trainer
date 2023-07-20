from torch.utils.data import Dataset


class FLAEPDataset(Dataset):
    def __init__(self, labels):
        self.x_data = labels

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        line = self.x_data[idx]
        stub = line.split(' ')
