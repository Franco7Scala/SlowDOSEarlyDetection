from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from src.support.utils import normalizeValues

class Mqtt(Dataset):
    def __init__(self, xy: pd.DataFrame):
        self.xy = xy.drop(['tcp.flags', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.conack.flags', 'mqtt.msg', 'mqtt.protoname'], axis="columns", inplace=True)
        self.xy = normalizeValues(xy)

        self.x = torch.tensor(self.xy.to_numpy()).float() #54 columns
        self.y = torch.tensor(self.xy[['target']].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy)