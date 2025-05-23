import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from src.datasets.Mqtt import Mqtt
from src.support import utils
from src.support.utils import get_base_dir

paths = [f"{get_base_dir()}/csvs/mqtt/train70.csv",
         f"{get_base_dir()}/csvs/mqtt/test30.csv"]

#-----dataframe-----#
dataframes = utils.readPaths(paths)
labels = utils.stringLabels(dataframes)
dataframes = utils.removeLabels(dataframes, labels, ["dos", "slowite", "legitimate"])
dataframes = utils.convertStringsSD(dataframes, labels)
train_df = dataframes[0]
train_dos_df, train_slowdos_df = utils.divideDataFrame(train_df)
train_slowdos_df["target"] = train_slowdos_df["target"].replace(2, 1)
test_df = dataframes[1]
test_dos_df, test_slowdos_df = utils.divideDataFrame(test_df)
test_slowdos_df["target"] = test_slowdos_df["target"].replace(2, 1)
train_slowdos_df, test_slowdos_df = utils.redistibuteSlowDosRows(train_slowdos_df, test_slowdos_df)
#-----dataframe-----#

#-----datasets-----#
dos_train = Mqtt(train_dos_df)
dos_test = Mqtt(test_dos_df)
slowdos_train = Mqtt(train_slowdos_df)
slowdos_test = Mqtt(test_slowdos_df)
#-----datasets-----#

#-----dataLoaders-----#
dos_train_loader = DataLoader(dos_train)
dos_test_loader = DataLoader(dos_test)
slowdos_train_loader = DataLoader(slowdos_train)
slowdos_test_loader = DataLoader(slowdos_test)

x_train_slowdos, y_train_slowdos = utils.convertDataLoaderToNumpy(slowdos_train_loader)
x_test_slowdos, y_test_slowdos = utils.convertDataLoaderToNumpy(slowdos_test_loader)
#-----dataLoaders-----#

input_size = train_slowdos_df.shape[1]
output_size = 2

dos_df = pd.concat([train_dos_df, test_dos_df])
slowdos_df = pd.concat([train_slowdos_df, test_slowdos_df])
dos_weights = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(dos_df["target"]), y=dos_df["target"]))
slodos_weights = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(slowdos_df["target"]), y=slowdos_df["target"]))

with (open(f"{get_base_dir()}/pickels/slowdos_test_loader.pkl", "wb")) as f:
    pickle.dump(slowdos_test_loader, f)

with (open(f"{get_base_dir()}/pickels/input_size.pkl", "wb")) as f:
    pickle.dump(input_size, f)

with (open(f"{get_base_dir()}/pickels/output_size.pkl", "wb")) as f:
    pickle.dump(output_size, f)

with (open(f"{get_base_dir()}/pickels/dos_weights.pkl", "wb")) as f:
    pickle.dump(dos_weights, f)

with (open(f"{get_base_dir()}/pickels/slowdos_weights.pkl", "wb")) as f:
    pickle.dump(slodos_weights, f)

with (open(f"{get_base_dir()}/pickels/dos_train_loader.pkl", "wb")) as f:
    pickle.dump(dos_train_loader, f)

with (open(f"{get_base_dir()}/pickels/slowdos_train_loader.pkl", "wb")) as f:
    pickle.dump(slowdos_train_loader, f)

with (open(f"{get_base_dir()}/pickels/x_train_slowdos.pkl", "wb")) as f:
    pickle.dump(x_train_slowdos, f)

with (open(f"{get_base_dir()}/pickels/y_train_slowdos.pkl", "wb")) as f:
    pickle.dump(y_train_slowdos, f)

with (open(f"{get_base_dir()}/pickels/x_test_slowdos.pkl", "wb")) as f:
    pickle.dump(x_test_slowdos, f)

with (open(f"{get_base_dir()}/pickels/y_test_slowdos.pkl", "wb")) as f:
    pickle.dump(y_test_slowdos, f)