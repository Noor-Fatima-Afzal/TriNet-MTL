import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, X, y1, y2, y3, seq_len=20000):
        self.seq_len = seq_len
        self.X = [self._pad_or_trunc(torch.tensor(x, dtype=torch.float32)) for x in X]
        self.y1 = torch.tensor(y1, dtype=torch.long)
        self.y2 = torch.tensor(y2, dtype=torch.long)
        self.y3 = torch.tensor(y3, dtype=torch.long)

    def _pad_or_trunc(self, x):
        if x.shape[0] > self.seq_len:
            return x[:self.seq_len]
        elif x.shape[0] < self.seq_len:
            padding = torch.zeros((self.seq_len - x.shape[0], x.shape[1]))
            return torch.cat([x, padding], dim=0)
        return x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx], self.y3[idx]
