import math

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Optional

from torch.utils.data import DataLoader

class PredictiveNN(nn.Module):
    def __init__(self, input_size, output_size, device, dropout: float, extend_input: Optional[bool] = False):
        super(PredictiveNN, self).__init__()
        self.device = device
        self.extend_input = extend_input
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        self.to(self.device)

    def forward(self, x):
        x = x.float()
        if self.extend_input:
            x = create_extended_input(x)
        logits = self.fully_connected_1(x)
        ret = torch.softmax(logits, dim=1)
        return ret

#-----train and test-----#
    def _train_epoch(self, train_loader, optimizer, criterion):
        self.train()
        loss_sum = 0
        for i, content in enumerate(train_loader):
            optimizer.zero_grad()
            target = content[1].to(self.device).view(-1).long()
            input = content[0].to(self.device)
            output = self(input)
            loss = criterion(output, target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            loss_sum += loss.item()

    def evaluate(self, loader, criterion):
        accuracy_am = AverageMeter('Accuracy', ':6.2f')
        precision_am = AverageMeter('Precision', ':6.2f')
        recall_am = AverageMeter('Recall', ':6.2f')
        f1_am = AverageMeter('F1', ':6.2f')
        all_preds = []
        all_targets = []
        self.eval()
        self.no_grad = True
        for i, (input, target) in enumerate(loader):
            input = input.to(self.device)
            target = target.to(self.device).view(-1).long()
            with torch.no_grad():
                output = self(input)
                loss = torch.sqrt(criterion(output, target))

            _, predicted = torch.max(output.data, 1)
            accuracy = accuracy_score(predicted.data.cpu(), target.data.cpu())
            accuracy_am.update(accuracy, input.size(0))
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted")
        f1 = f1_score(all_targets, all_preds, average="weighted")

        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        self.no_grad = False

        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg

    def fit(self, epochs, optimizer, criterion, train_loader, test_loader: Optional[DataLoader] = None):
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for epoch in range(epochs):
            self._train_epoch(train_loader, optimizer, criterion)
            if test_loader is not None:
                new_accuracy, new_precision, new_recall, new_f1 = self.evaluate(test_loader, criterion)
                if new_accuracy > accuracy:
                    accuracy, precision, recall, f1 = new_accuracy, new_precision, new_recall, new_f1
                    self.save("best_accuracy_scoring_predictive_nn.pt")
            print("epoch:", epoch)
        print("Finished training predictive NN!")
        if test_loader is not None:
            print("Final results:")
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
# -----train and test-----#

    def save(self, path):
        torch.save(self.state_dict(), path)

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


def create_extended_input(raw_input_layer):
    extended_features = []
    range_values = [8]

    extended_features.append(raw_input_layer)

    #one_minus_i = Lambda(lambda x: 1 - torch.clip(x, 0, 1))(raw_input_layer)
    #extended_features.append(one_minus_i)

    # power
    for v in range_values:
        power_i = LambdaLayer(lambda x: x ** v)(raw_input_layer)
        extended_features.append(power_i)

    # root
    for v in range_values:
        root_i = LambdaLayer(lambda x: torch.clip(x, 0, 1) ** (1 / v))(raw_input_layer)
        extended_features.append(root_i)

    #sin and 1-cos
    #sin_i = Lambda(lambda x: torch.sin(math.pi * torch.clip(x, 0, 1)))(raw_input_layer)
    #extended_features.append(sin_i)
    #one_minus_cos_i = Lambda(lambda x: 1 - torch.cos(math.pi * torch.clip(x, 0, 1)))(raw_input_layer)
    #extended_features.append(one_minus_cos_i)

    # other extensions
    log_i = LambdaLayer(lambda x: torch.log(torch.clip(x, 0, 1) + 1) / math.log(2))(raw_input_layer)
    extended_features.append(log_i)
    #one_minus_inv_log_i = Lambda(lambda x: 1 - torch.log(torch.clip(-x, 0, 1) + 2) / math.log(2))(raw_input_layer)
    #extended_features.append(one_minus_inv_log_i)
    exp_i = LambdaLayer(lambda x: torch.exp(x - 1))(raw_input_layer)
    extended_features.append(exp_i)
    #one_minus_exp_i = Lambda(lambda x: 1 - torch.exp(-x))(raw_input_layer)
    #extended_features.append(one_minus_exp_i)

    # improved input
    return torch.concat(extended_features)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)