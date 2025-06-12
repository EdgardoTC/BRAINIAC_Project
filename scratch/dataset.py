import torch
from torch.utils.data import Dataset
import numpy as np

class BrainDataset(Dataset):
    def __init__(self, scan_paths, labels):
        self.scan_paths = scan_paths
        self.labels = labels

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        x = np.load(self.scan_paths[idx])
        x = torch.tensor(x).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
