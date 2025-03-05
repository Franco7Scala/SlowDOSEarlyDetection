import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import Cicids
from src.nets import NeuralNetwork
from src.support import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "C:/Users/black/OneDrive/Desktop/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
epochs = 50

dataset = Cicids.Cicids2017(path)

train_set, test_set = random_split(dataset, [0.7, 0.3])
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, shuffle=True)

model = NeuralNetwork.NeuralNetwork(79, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    utils.train(train_loader, model, optimizer, criterion, device)
    if epoch % 10 == 0:
        accuracy, precision, recall, f1 = utils.test(test_loader, model, criterion, device)
        print("epoch:", epoch)
        print("accuracy: "+accuracy,"precision: "+precision,"recall: "+recall,"f1: "+f1)