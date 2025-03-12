import torch
from torch.utils.data import Dataset

from src.support import utils

class Cicids2017(Dataset):
    def __init__(self, xy):
        xy = xy.drop([' Destination Port'], axis="columns")
        xy = utils.normalizeValues(xy)

        self.x = torch.tensor(xy.to_numpy()).float() #78 columns
        self.y = torch.tensor(xy[[' Label']].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)