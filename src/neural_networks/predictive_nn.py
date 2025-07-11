import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_auc_score
from typing import Optional
from torch.utils.data import DataLoader


class PredictiveNN(nn.Module):
    def __init__(self, input_size, output_size, device, dropout: float):
        super(PredictiveNN, self).__init__()
        self.device = device
        layer_size = 32
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(layer_size),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(layer_size),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(layer_size),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(layer_size),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(inplace=True),
        )
        self.fully_connected_2 = nn.Sequential(
            nn.BatchNorm1d(layer_size),
            nn.Dropout(dropout),
            nn.Linear(layer_size, output_size)
        )
        self.to(self.device)

    def forward(self, x):
        encoded = self.encode(x)
        logits = self.fully_connected_2(encoded)
        ret = torch.softmax(logits, dim=1)
        return ret

    def encode(self, x):
        x = x.float()
        return self.fully_connected_1(x)

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
        auc = roc_auc_score(y_true=all_targets, y_score=all_preds)
        cr = classification_report(all_targets, all_preds, target_names=["Benign", "SlowDoS"])


        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        self.no_grad = False

        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg, auc, cr

    def fit(self, epochs, optimizer, criterion, train_loader, test_loader: Optional[DataLoader] = None):
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for epoch in tqdm(range(epochs)):
            self._train_epoch(train_loader, optimizer, criterion)
            if test_loader is not None:
                new_accuracy, new_precision, new_recall, new_f1 = self.evaluate(test_loader, criterion)
                if new_accuracy > accuracy:
                    accuracy, precision, recall, f1 = new_accuracy, new_precision, new_recall, new_f1
                    self.save("best_accuracy_scoring_predictive_nn.pt")

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


