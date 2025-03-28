import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import src.datasets.Cicids as Cicids2017


#-----train and test-----#
def train(train_loader, network, optimizer, criterion, device):
    network.train()
    loss_sum = 0
    for i, content in enumerate(train_loader):
        optimizer.zero_grad()
        target = content[1].to(device).view(-1).long()
        input = content[0].to(device)
        output = network(input)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()
        loss_sum += loss.item()

def test(test_loader, network, criterion, device):
    accuracy_am = AverageMeter('Accuracy', ':6.2f')
    precision_am = AverageMeter('Precision', ':6.2f')
    recall_am = AverageMeter('Recall', ':6.2f')
    f1_am = AverageMeter('F1', ':6.2f')
    all_preds = []
    all_targets = []
    network.eval()
    network.no_grad = True
    for i, (input, target) in enumerate(test_loader):
        input = input.to(device)
        target = target.to(device).view(-1).long()
        with torch.no_grad():
            output = network(input)
            loss = torch.sqrt(criterion(output, target))

        _, predicted = torch.max(output.data, 1)
        accuracy = accuracy_score(predicted.data.cpu(), target.data.cpu())
        accuracy_am.update(accuracy, input.size(0))
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    precision = precision_score(all_targets, all_preds, average="weighted") #TOFIX for MultiClassModel
    recall = recall_score(all_targets, all_preds, average="weighted") #TOFIX for MultiClassModel
    f1 = f1_score(all_targets, all_preds, average="weighted") #TOFIX for MultiClassModel

    precision_am.update(precision)
    recall_am.update(recall)
    f1_am.update(f1)

    network.no_grad = False

    return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg

#da: PlayItStraiht di Francesco Scala
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
#-----train and test-----#

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

def splitDataset(dataset: Cicids2017) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    x = dataset.x
    y = dataset.y
    sss = StratifiedShuffleSplit(train_size=0.7, test_size=0.3, random_state=50)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    train = list(zip(x_train, y_train))
    test = list(zip(x_test, y_test))
    return train, test
#-----datasets utils-----#