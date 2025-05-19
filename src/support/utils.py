import numpy as np
import pandas as pd
import torch
import os

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset, TensorDataset
from collections import defaultdict
import random


def get_base_dir():
    #return "C:/Users/black/OneDrive/Desktop/Università/Tesi/cicids2017/csvs\MachineLearningCSV"
    return "/home/scala/projects/SlowDosDetection/data"


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

def convertStringsSD(dataFrames: list[pd.DataFrame], labels: list[str]) -> list[pd.DataFrame]:
    ret = []
    for df in dataFrames:
        for string in labels:
            if string == "DDoS":
                df = df.replace(string, 1)
            elif string == "DoS Slowhttptest" or string == "DoS slowloris":
                df = df.replace(string, 2)
            else:
                df = df.replace(string, 0)
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

def stratifiedSampling(dataset, target: torch.Tensor, train_size: float) -> (Subset, Subset):
    train_index, validation_index = train_test_split(np.arange(len(dataset)), train_size=train_size, random_state=999, shuffle=True, stratify=target)

    train_dataset = Subset(dataset, train_index)
    validation_dataset = Subset(dataset, validation_index)
    return train_dataset, validation_dataset

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
        if "Tuesday" in path:
            tuesdayPaths.append(path)
        if "Wednesday" in path:
            wednesdayPaths.append(path)
        if "Thursday" in path:
            thursdayPaths.append(path)
        if "Friday" in path:
            fridayPaths.append(path)
    ret.append(len(pd.concat(readPaths(tuesdayPaths))))
    ret.append(len(pd.concat(readPaths(wednesdayPaths))))
    ret.append(len(pd.concat(readPaths(thursdayPaths))))
    ret.append(len(pd.concat(readPaths(fridayPaths))))
    return ret

def splitWeekDaysDatasets(lengths: list[int], dataset: Dataset) -> (Dataset, Dataset, Dataset, Dataset):
    ret = []
    start = 0
    for length in lengths:
        end = start + length - 1
        ret.append(Subset(dataset, range(start, end)))
        start = end
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

def extract_xy_from_concat(dataset: ConcatDataset):
    x_list = []
    y_list = []

    for d in dataset.datasets:
        # Se è un Subset
        if isinstance(d, Subset):
            base_ds = d.dataset
            indices = d.indices
            x_list.append(base_ds[0][indices])
            y_list.append(base_ds[1][indices])
        # Se è un TensorDataset diretto
        elif isinstance(d, TensorDataset):
            x_list.append(d.tensors[0])
            y_list.append(d.tensors[1])
        else:
            raise TypeError(f"Unsupported dataset type: {type(d)}")

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)

    assert len(x) == len(y), f"x and y have different lengths: {len(x)} != {len(y)}"

    return x, y

def duplicateClass(dataset, class_to_dup: float, times_to_dup: int):
    x_new = []
    y_new = []

    for i in range(len(dataset.x)):
        x_new.append(dataset.x[i].unsqueeze(0))
        y_new.append(dataset.y[i].unsqueeze(0))
        if dataset.y[i].item() == class_to_dup:
            for _ in range(times_to_dup):
                x_new.append(dataset.x[i].unsqueeze(0))
                y_new.append(dataset.y[i].unsqueeze(0))

    dataset.x = torch.cat(x_new, dim=0)
    dataset.y = torch.cat(y_new, dim=0)

def createCustomSplit(dataset, K: int, H: int) -> (torch.Tensor, torch.Tensor):
    num_classes = 14
    n_per_class = K // num_classes
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[int(label)].append(idx)

    for cls in range(0, num_classes + 1):
        needed = n_per_class + (H if cls == 0 else 0)
        if len(class_indices[cls]) < needed:
            raise ValueError(f"Classe {cls} ha solo {len(class_indices[cls])} esempi, ma ne servono {needed}")

    selected_indices = []
    used_in_target_class = set()

    for cls in range(0, num_classes + 1):
        indices = class_indices[cls]
        random.shuffle(indices)
        count = n_per_class + (H if cls == 0 else 0)
        selected = indices[:count]
        selected_indices.extend(selected)

        if cls == 0:
            used_in_target_class.update(selected[:n_per_class])

    data_list = []
    label_list = []
    for idx in selected_indices:
        x, y = dataset[idx]
        data_list.append(x)
        label_list.append(y)

    return torch.stack(data_list), torch.tensor(label_list)

def createCustomSplitSlowDos(dataset, y, indices, class_0_percentage: float, class_1_percentage: float) -> (Subset, Subset):
    original_indices = np.array(indices)
    labels = np.array(y)

    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]

    np.random.seed(42)
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    split_0 = int(len(class_0_indices) * class_0_percentage)
    train_0 = class_0_indices[:split_0]
    test_0 = class_0_indices[split_0:]

    split_1 = int(len(class_1_indices) * class_1_percentage)
    train_1 = class_1_indices[:split_1]
    test_1 = class_1_indices[split_1:]

    train_indices = original_indices[np.concatenate([train_0, train_1])]
    test_indices = original_indices[np.concatenate([test_0, test_1])]

    train = Subset(dataset, train_indices)
    test = Subset(dataset, test_indices)

    return train, test

def removeLabels(dataframes: list[pd.DataFrame], labels: list[str], labels_to_keep: list[str]) -> list[pd.DataFrame]:
    ret = []
    for df in dataframes:
        for string in labels:
            if string not in labels_to_keep:
                df = df[df[" Label"] != string]
        ret.append(df)
    return ret

def divideDataFrame(dataframe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_DDoS, df_slowDoS = pd.DataFrame(), pd.DataFrame()
    benign = dataframe[dataframe[" Label"] == 0] #only benign rows
    benign_half = len(benign) // 2 #half-length of benign rows
    ddos = dataframe[dataframe[" Label"] == 1] #only ddos rows
    slow_dos = dataframe[dataframe[" Label"] == 2] #only slow dos rows
    df_DDoS = pd.concat([df_DDoS, benign.iloc[:benign_half], ddos], ignore_index=True)
    df_slowDoS = pd.concat([df_slowDoS, benign.iloc[benign_half + 1:], slow_dos], ignore_index=True)
    return df_DDoS, df_slowDoS

def divideDataset(dataset):
    benign_idx = []
    ddos_idx = []
    slowdos_idx = []

    for i in range(len(dataset)):
        _, y = dataset[i]
        label = y.item() if isinstance(y, torch.Tensor) else y
        if label == 0:
            benign_idx.append(i)
        elif label == 1:
            ddos_idx.append(i)
        elif label == 2:
            slowdos_idx.append(i)

    half = len(benign_idx) // 2
    benign_half1 = benign_idx[:half]
    benign_half2 = benign_idx[half:]

    ddos_subset = Subset(dataset, benign_half1 + ddos_idx)
    slowdos_subset = Subset(dataset, benign_half2 + slowdos_idx)

    return ddos_subset, slowdos_subset
#-----datasets utils-----#