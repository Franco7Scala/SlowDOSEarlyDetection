import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.datasets import Cicids
from src.nets import NeuralNetwork
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
dataframes = utils.readPaths(paths)
labels = utils.stringLabels(dataframes)
dataframes = utils.convertStringsMC(dataframes, labels)
dataframe = pd.concat(dataframes)
#-----DataFrame-----#

weights = utils.assigngWeights(dataframe)

dataset = Cicids.Cicids2017(dataframe)

input_size = dataset.x.shape[1]

output_size = len(labels)

batch_size = 500

#-----Train, Validation and Test DataLoaders-----#
x_main, x_test, y_main, y_test = utils.splitDataset(dataset.x, dataset.y, 0.7, 0.3)
x_train, x_validation, y_train, y_validation = utils.splitDataset(x_main, y_main, 0.8, 0.2)

train = TensorDataset(x_train, y_train)
validation = TensorDataset(x_validation, y_validation)
test = TensorDataset(x_test, y_test)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
#-----Train, Validation and Test DataLoaders-----#

#-----MultiClassModel-----#
model = NeuralNetwork.NeuralNetwork(input_size, output_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

epochs = 150
print("Starting Training...")
for epoch in range(epochs):
    utils.train(train_loader, model, optimizer, criterion, device)
    if epoch % 10 == 0 or epoch == epochs - 1:
        accuracy, precision, recall, f1 = utils.evaluate(validation_loader, model, criterion, device)
        print("epoch:", epoch)
        print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
print("Finished Training!")
print("Starting Testing...")
accuracy, precision, recall, f1 = utils.evaluate(test_loader, model, criterion, device)
print("Test results:")
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
#-----MultiClassModel-----#