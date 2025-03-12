import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#-----train and test-----#
def train(train_loader, network, optimizer, criterion, device):
    loss_sum = 0
    for i, content in enumerate(train_loader):
        optimizer.zero_grad()
        target = content[1].to(device).view(-1)
        input = content[0].to(device)
        if (input.isnan().any()):
            print("input is nan")
        output = network(input)
        if (output.isnan().any()):
            print("output is nan")
        loss = torch.sqrt(criterion(output, target))
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
        target = target.to(device).view(-1)
        with torch.no_grad():
            output = network(input)
            loss = torch.sqrt(criterion(output, target))
        accuracy = accuracy_score(output.data.cpu(), target.data.cpu())
        accuracy_am.update(accuracy, input.size(0))

        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

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
        label = df[' Label'].drop_duplicates()
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

def convertStrings(dataFrames: [pd.DataFrame], labels: []) -> []:
    ret = []
    value = 0.1
    for df in dataFrames:
        for string in labels:
            df = df.replace(string, value)
            value += 0.1
        ret.append(df)
    return ret

def normalizeValues(dataFrame: pd.DataFrame) -> pd.DataFrame:
    ret = dataFrame
    for column in ret.columns:
        if column != ' Label':
            mean_value = ret[column].mean(axis=0)
            ret[column] = ret[column].replace(0, mean_value)
            ret[column] = ret[column].fillna(mean_value)
    return ret
#-----datasets utils-----#