import torch
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np

from src.datasets import Cicids
from src.datasets import Randomset
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
dataFrame = pd.concat(dataFrames)

dataset = Cicids.Cicids2017(dataFrame)

input_size = dataset.__len__() - 1

train, test = utils.splitDataset(dataset)

train_loader = DataLoader(train, batch_size=100, shuffle=True)
test_loader = DataLoader(test, batch_size=100, shuffle=True)

#-----random dataset-----#
'''
test = pd.DataFrame(np.random.rand(300000, 78), columns=list(range(78)), dtype=float)
test_dataset = Randomset.Randomset(test)
test_train, test_test = random_split(test_dataset, [0.7, 0.3])
test_train_loader = DataLoader(test_train, batch_size=100, shuffle=True)
test_test_loader = DataLoader(test_test, batch_size=100, shuffle=True)
'''
#-----random dataset-----#


model = NeuralNetwork.NeuralNetwork(input_size, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

criterion = torch.nn.MSELoss()

epochs = 50
for epoch in range(epochs):
    utils.train(train_loader, model, optimizer, criterion, device)
    if epoch % 10 == 0:
        accuracy, precision, recall, f1 = utils.test(test_loader, model, criterion, device)
        print("epoch:", epoch)
        print("accuracy: "+accuracy,"precision: "+precision,"recall: "+recall,"f1: "+f1)