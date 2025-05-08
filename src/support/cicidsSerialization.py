import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset, random_split

from src.datasets import Cicids
from src.support import utils

paths = ["C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"]

print("Processing dateset...")
#-----DataFrame-----#
dataframes = utils.readPaths(paths)
labels = utils.stringLabels(dataframes)
dataframes = utils.convertStringsMC(dataframes, labels)
dataframe = pd.concat(dataframes)
#-----DataFrame-----#

week_days_lengths = utils.getWeekDaysLengths(paths)
'''
week_days_lengths[1] += 11 * 8
week_days_lengths[2] += (36 + 21) * 3
'''
weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataframe[' Label']), y=dataframe[' Label'])
weights_tensor = torch.Tensor(weights)

dataset = Cicids.Cicids2017(dataframe, True)
'''
utils.duplicateClass(dataset, 7, 8)
utils.duplicateClass(dataset, 11, 3)
utils.duplicateClass(dataset, 8, 3)
'''
week_days_datasets = utils.splitWeekDaysDatasets(week_days_lengths, dataset) #splitting dataset into single-days subset
print("Done!")

input_size = dataset.x.shape[1]

output_size = len(labels)

batch_size = 512

#-----Train, Validation and Test DataLoaders-----#
print("Creating train, validation and test dataloaders...")
#trains = []
tests = []
for subset in week_days_datasets:
    x = subset.dataset.x[subset.indices]
    y = subset.dataset.y[subset.indices]
    x_train, x_test, y_train, y_test = utils.splitDataset(x, y, 0.7, 0.3)
    #trains.append(TensorDataset(x_train, y_train))
    tests.append(TensorDataset(x_test, y_test))
#train = ConcatDataset(trains)

x_train, y_train = utils.createCustomTrainset(dataset, 80, 240)
y_count = np.unique(y_train, return_counts=True)
train = TensorDataset(x_train, y_train)
#-----Partition train set-----#
'''
x, y = utils.extract_xy_from_concat(train)

train, _ = utils.stratifiedSampling(train, y, 0.0125)
'''
#-----Partition train set-----#

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loaders = []
for test in tests:
    test_loaders.append(DataLoader(test, batch_size=batch_size, shuffle=True))

x_train, y_train = utils.convertDataLoaderToNumpy(train_loader) #numpy for other models
print("Done!")
#-----Train, Validation and Test DataLoaders-----#

with (open("C:/Coding/PyCharm Projects/src/support/files/train_loader.pkl", "wb")) as f:
    pickle.dump(train_loader, f)

with (open("C:/Coding/PyCharm Projects/src/support/files/test_loaders.pkl", "wb")) as f:
    pickle.dump(test_loaders, f)

with (open("C:/Coding/PyCharm Projects/src/support/files/x_train.pkl", "wb")) as f:
    pickle.dump(x_train, f)

with (open("C:/Coding/PyCharm Projects/src/support/files/y_train.pkl", "wb")) as f:
    pickle.dump(y_train, f)

with (open("C:/Coding/PyCharm Projects/src/support/files/input_size.pkl", "wb")) as f:
    pickle.dump(input_size, f)

with (open("C:/Coding/PyCharm Projects/src/support/files/output_size.pkl", "wb")) as f:
    pickle.dump(output_size, f)

with (open("C:/Coding/PyCharm Projects/src/support/files/weights_tensor.pkl", "wb")) as f:
    pickle.dump(weights_tensor, f)