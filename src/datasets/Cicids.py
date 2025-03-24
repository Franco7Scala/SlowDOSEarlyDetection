import pandas as pd
import torch
from torch.utils.data import Dataset
from src.support.utils import removeCollinearFeatures, normalizeValues

from src.support import utils

class Cicids2017(Dataset):
    def __init__(self, xy: pd.DataFrame):
        self.xy = xy.drop([' Destination Port'], axis="columns")
        self.xy = normalizeValues(xy)
        self.xy = removeCollinearFeatures(xy, 0.95)

        self.x = torch.tensor(xy.to_numpy()).float() #54 columns
        self.y = torch.tensor(xy[[' Label']].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy.columns)