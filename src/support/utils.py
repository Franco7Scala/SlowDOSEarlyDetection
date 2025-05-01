import numpy as np
import pandas as pd
import torch
import random
import os

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset, DataLoader


#-----datasets utils-----#
def stringLabels(dataFrames: list[pd.DataFrame]) -> list[str]:
    ret = []
    for df in dataFrames:
        label = df[' Label'].unique()
        for string in label:
            if string not in ret:
                ret.append(string)
    return ret

def readPaths(paths: list[str]) -> list[pd.DataFrame]:
    ret = []
    for path in paths:
        df = pd.read_csv(path)
        ret.append(df)
    return ret

def convertStringsTC(dataFrames: list[pd.DataFrame], labels: list[str]) -> list[pd.DataFrame]:
    ret = []
    for df in dataFrames:
        for string in labels:
            if string == "BENIGN":
                df = df.replace(string, 0)
            else:
                df = df.replace(string, 1)
        ret.append(df)
    return ret

def convertStringsMC(dataFrames: list[pd.DataFrame], labels: list[str]) -> list[pd.DataFrame]:
    ret = []
    value = 0
    for df in dataFrames:
        for string in labels:
            if string == "BENIGN":
                df = df.replace(string, 0)
            else:
                if string in df.values:
                    value += 1
                    df = df.replace(string, value)
        ret.append(df)
    return ret

def normalizeValues(dataFrame: pd.DataFrame) -> pd.DataFrame:
    ret = dataFrame
    for column in ret.columns:
        if column != ' Label':
            ret[column] = ret[column].replace(np.inf, 0) #replace inf with zero
            for value in ret[column].unique():
                if value < 0:
                    ret[column] = ret[column].replace(value, 0) #replace negatives with zero
            mean_value = ret[column].mean(axis=0)
            ret[column] = ret[column].replace(0, mean_value) #replace 0 with mean value of the column
            ret[column] = ret[column].fillna(mean_value) #replace na with mean value of the column
    return ret

def removeCollinearFeatures(dataFrame: pd.DataFrame, threshold) -> pd.DataFrame:
    ret = dataFrame
    corr_matrix = dataFrame.corr(method='pearson', min_periods=5, numeric_only=True)
    iters = range(len(corr_matrix.columns) - 1)
    drop_columns = []
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            val = abs(item.values)
            if val >= threshold:
                drop_columns.append(col.values[0])
    drop_columns = list(set(drop_columns))
    ret = ret.drop(columns=drop_columns)
    return ret

def splitDataset(data: torch.Tensor, target: torch.Tensor, size1: float, size2: float) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    global y_test, y_train, x_test, x_train
    sss = StratifiedShuffleSplit(train_size=size1, test_size=size2, random_state=50)
    for train_index, test_index in sss.split(data, target):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
    return x_train, x_test, y_train, y_test

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getWeekDaysLengths(paths: list[str]) -> list[int]:
    ret = []
    fridayPaths, thursdayPaths, tuesdayPaths, wednesdayPaths = [], [], [], []
    for path in paths:
        if "Thursday" in path:
            thursdayPaths.append(path)
        if "Wednesday" in path:
            wednesdayPaths.append(path)
        if "Tuesday" in path:
            tuesdayPaths.append(path)
        if "Friday" in path:
            fridayPaths.append(path)
    ret.append(len(pd.concat(readPaths(thursdayPaths))))
    ret.append(len(pd.concat(readPaths(wednesdayPaths))))
    ret.append(len(pd.concat(readPaths(tuesdayPaths))))
    ret.append(len(pd.concat(readPaths(fridayPaths))))
    return ret

def splitWeekDaysDatasets(lengths: list[int], dataset: Dataset) -> (Dataset, Dataset, Dataset, Dataset):
    ret = []
    start = 0
    for length in lengths:
        end = start + length
        ret.append(Subset(dataset, range(start, end)))
        start = end + 1
    return ret

def convertDataLoaderToNumpy(loader: DataLoader) -> (np.ndarray, np.ndarray):
    x_list = []
    y_list = []
    for batch in loader:
        input, labels = batch
        x_list.append(input.numpy())
        y_list.append(labels.numpy())
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    return x, y
#-----datasets utils-----#