import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import compute_class_weight


#-----datasets utils-----#
def stringLabels(dataFrames: [pd.DataFrame]) -> []:
    ret = []
    for df in dataFrames:
        label = df[' Label'].unique()
        for string in label:
            if string not in ret:
                ret.append(string)
    return ret

def readPaths(paths: []) -> []:
    ret = []
    for path in paths:
        df = pd.read_csv(path)
        ret.append(df)
    return ret

def convertStringsTC(dataFrames: [pd.DataFrame], labels: []) -> []:
    ret = []
    for df in dataFrames:
        for string in labels:
            if string == "BENIGN":
                df = df.replace(string, 0)
            else:
                df = df.replace(string, 1)
        ret.append(df)
    return ret

def convertStringsMC(dataFrames: [pd.DataFrame], labels: []) -> []:
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
#-----datasets utils-----#