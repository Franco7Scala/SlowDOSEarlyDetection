import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class ConcatenatedPredictiveVAE(nn.Module):

    def __init__(self, model1, model2, input_size, output_size, device):
        super(ConcatenatedPredictiveVAE, self).__init__()
        self.device = device
        self.model1 = model1
        self.model2 = model2
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.to(self.device)

    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x2 = x2[2]
        x2 = torch.Tensor(x2)
        x = torch.cat((x1, x2), dim=1)
        logits = self.fully_connected_1(x)
        return logits

# -----train and test-----#
    def _train_epoch(self, train_loader, optimizer, criterion):
        self.train()

        #-----freeze model1 and model2-----#
        self.model1.eval()
        self.model2.eval()
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = True
        # -----freeze model1 and model2-----#

        loss_sum = 0
        for i, content in enumerate(train_loader):
            optimizer.zero_grad()
            target = content[1].to(self.device).view(-1).long()
            input1 = content[0].to(self.device)
            input2 = content[0].to(self.device)
            output = self(input1, input2)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
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

        precision = precision_score(all_targets, all_preds, average="weighted")
        recall = recall_score(all_targets, all_preds, average="weighted")
        f1 = f1_score(all_targets, all_preds, average="weighted")

        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        self.no_grad = False

        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg

    def fit(self, epochs, train_loader, test_loader, optimizer, criterion):
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for epoch in range(epochs):
            self._train_epoch(train_loader, optimizer, criterion)
            accuracy, precision, recall, f1 = self.evaluate(test_loader, criterion)
            print("epoch:", epoch)
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
        print("Finished training!")
        print("Final results:")
        print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
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