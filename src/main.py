import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd

from src.datasets import Cicids
from src.nets import NeuralNetwork
from src.support import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
paths = ["C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"]

dataFrames = utils.readPaths(paths)
labels = utils.stringLabels(dataFrames)
dataFrames = utils.convertStrings(dataFrames, labels)

datasets = []
for df in dataFrames:
    datasets.append(Cicids.Cicids2017(df))

train_sets, test_sets = [], []
for dataset in datasets:
    train, test = random_split(dataset, [0.7, 0.3])
    train_sets.append(train)
    test_sets.append(test)

train_set = torch.utils.data.ConcatDataset(train_sets)
test_set = torch.utils.data.ConcatDataset(test_sets)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

model = NeuralNetwork.NeuralNetwork(78, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

criterion = torch.nn.MSELoss()

epochs = 50
for epoch in range(epochs):
    utils.train(train_loader, model, optimizer, criterion, device)
    if epoch % 10 == 0:
        accuracy, precision, recall, f1 = utils.test(test_loader, model, criterion, device)
        print("epoch:", epoch)
        print("accuracy: "+accuracy,"precision: "+precision,"recall: "+recall,"f1: "+f1)