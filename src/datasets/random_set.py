import torch
from torch.utils.data import Dataset

class Randomset(Dataset):
    def __init__(self, xy):

        self.x = torch.tensor(xy.to_numpy()).float()  # 78 columns
        self.y = torch.tensor(xy[0].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
