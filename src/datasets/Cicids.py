import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Cicids2017(Dataset):
    def __init__(self, path):
        xy = pd.read_csv(path)
        xy = convertStrings(xy)
        xy = xy.fillna(0)
        self.x = torch.tensor(xy.to_numpy()) #79 columns
        self.y = torch.tensor(xy[[' Label']].to_numpy())

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def convertStrings(dataFrame: pd.DataFrame) -> pd.DataFrame:
    ret = dataFrame
    string_columns = ret[' Label']
    list = []
    for string in string_columns:
        if (string not in list):
            list.append(string)
    i = 0
    for string in list:
        ret = ret.replace(string, i)
        i += 1
    return ret