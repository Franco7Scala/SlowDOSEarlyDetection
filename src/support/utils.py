import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def train(train_loader, network, optimizer, criterion, arg):
    loss_sum = 0
    for i, content in enumerate(train_loader):
        optimizer.zero_grad()
        target = content[1].to(arg.device)
        input = content[0].to(arg.device)
        output = network(input)
        loss = criterion(output, target).mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

def test(test_loader, network, criterion, arg):
    accuracy_am = AverageMeter('Accuracy', ':6.2f')
    precision_am = AverageMeter('Precision', ':6.2f')
    recall_am = AverageMeter('Recall', ':6.2f')
    f1_am = AverageMeter('F1', ':6.2f')
    all_preds = []
    all_targets = []
    network.eval()
    network.no_grad = True
    for i, (input, target) in enumerate(test_loader):
        target = input.to(arg.device)
        input = target.to(arg.device)
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()
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