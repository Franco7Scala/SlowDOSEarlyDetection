import numpy as np
import pandas as pd
import torch
import random
import os

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset, TensorDataset

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

def duplicateClass(dataset, class_to_dup: float, times_to_dup):
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

from collections import defaultdict
import random
from torch.utils.data import Subset

def createCustomTrainset(dataset, K, H):
    num_classes = 14
    n_per_class = K // num_classes
    class_indices = defaultdict(list)

    # Raccogli tutti gli indici organizzati per classe
    for idx, (_, label) in enumerate(dataset):
        class_indices[int(label)].append(idx)

    # Controlla che ogni classe abbia abbastanza esempi
    for cls in range(0, num_classes + 1):
        needed = n_per_class + (H if cls == 0 else 0)
        if len(class_indices[cls]) < needed:
            raise ValueError(f"Classe {cls} ha solo {len(class_indices[cls])} esempi, ma ne servono {needed}")

    # Seleziona gli indici per ciascuna classe
    selected_indices = []
    used_in_target_class = set()

    for cls in range(0, num_classes + 1):
        indices = class_indices[cls]
        random.shuffle(indices)
        count = n_per_class + (H if cls == 0 else 0)
        selected = indices[:count]
        selected_indices.extend(selected)

        # Salva quali hai già usato nella target class (per evitare duplicati se vuoi)
        if cls == 0:
            used_in_target_class.update(selected[:n_per_class])

    # Costruisci liste finali di dati e label
    data_list = []
    label_list = []
    for idx in selected_indices:
        x, y = dataset[idx]
        data_list.append(x)
        label_list.append(y)

    # Se vuoi, puoi anche restituire tensori concatenati
    return torch.stack(data_list), torch.tensor(label_list)
#-----datasets utils-----#