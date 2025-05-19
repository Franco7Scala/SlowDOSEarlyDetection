from typing import Optional

from tqdm import tqdm
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader


class ConcatenatedPredictiveVAE(nn.Module):

    def __init__(self, model1, model2, model3, input_size, output_size, device):
        super(ConcatenatedPredictiveVAE, self).__init__()
        self.device = device
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.to(self.device)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3.encode(x)
        x = torch.cat((x1, x2, x3), dim=1)
        logits = self.fully_connected_1(x)
        return logits

# -----train and test-----#
    def _train_epoch(self, train_loader, optimizer, criterion):
        self.train()

        #-----freeze model1, model2 and model3-----#
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        for param in self.model3.parameters():
            param.requires_grad = False
        # -----freeze model1, model2 and model3-----#

        loss_sum = 0
        count = 0

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
            count += 1

        return loss_sum / count

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
            input1 = input.to(self.device)
            input2 = input.to(self.device)
            target = target.to(self.device).view(-1).long()
            with torch.no_grad():
                output = self(input1, input2)
                loss = torch.sqrt(criterion(output, target))

            _, predicted = torch.max(output.data, 1)
            accuracy = accuracy_score(predicted.data.cpu(), target.data.cpu())
            accuracy_am.update(accuracy, input.size(0))
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        cr = classification_report(all_targets, all_preds, target_names=["Benign", "SlowDoS"])

        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        self.no_grad = False

        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg, cr

    def fit(self, epochs, optimizer, criterion, train_loader, test_loader: Optional[DataLoader] = None):
        train_losses_per_epoch = []
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for epoch in tqdm(range(epochs)):
            avg_loss = self._train_epoch(train_loader, optimizer, criterion)
            train_losses_per_epoch.append(avg_loss)
            if test_loader is not None:
                accuracy, precision, recall, f1, cr = self.evaluate(test_loader, criterion)

        print("Finished training CPVAE!")
        if test_loader is not None:
            print("Final results:")
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
        self.plotLoss(train_losses_per_epoch)

    def plotLoss(self, loss):
        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

# -----train and test-----#

# da: PlayItStraiht di Francesco Scala
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