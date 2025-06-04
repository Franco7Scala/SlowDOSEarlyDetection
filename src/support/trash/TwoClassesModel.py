import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.datasets import cicids
from src.nets import PredictiveNN
from src.support import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paths = ["C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        #"C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",#
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
        "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"]

#-----DataFrame-----#
dataFrames = utils.readPaths(paths)
labels = utils.stringLabels(dataFrames)
dataFrames = utils.convertStringsTC(dataFrames, labels)
dataFrame = pd.concat(dataFrames)
#-----DataFrame-----#

dataset = Cicids.Cicids2017(dataFrame)

input_size = dataset.x.shape[1]

batch_size = 500

x_train, x_test, y_train, y_test = utils.splitDataset(dataset.x, dataset.y, 0.7, 0.3)

train = TensorDataset(x_train, y_train)
test = TensorDataset(x_test, y_test)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

#-----TwoClassesModel-----#
model = PredictiveNN.NeuralNetwork(input_size, 2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

criterion = torch.nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    utils.train(train_loader, model, optimizer, criterion, device)
    if epoch % 10 == 0 or epoch == 49:
        accuracy, precision, recall, f1 = utils.test(test_loader, model, criterion, device)
        print("epoch:", epoch)
        print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
#-----TwoClassesModel-----#