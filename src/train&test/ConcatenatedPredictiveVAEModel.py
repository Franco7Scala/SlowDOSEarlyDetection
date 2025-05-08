import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time

from src.datasets import Cicids
from src.support import utils
from src.nets.PredictiveNN import PredictiveNN
from src.nets.VAENN import VAENN
from src.nets.ConcatenatedPredictiveVAENN import ConcatenatedPredictiveVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

paths = ["C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        #"C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",#
        "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
        "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"]

#-----DataFrame-----#
print("Processing dateset...")
dataframes = utils.readPaths(paths)
labels = utils.stringLabels(dataframes)
dataframes = utils.convertStringsMC(dataframes, labels)
dataframe = pd.concat(dataframes)
#-----DataFrame-----#

weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataframe[' Label']), y=dataframe[' Label'])
weights_tensor = torch.Tensor(weights)

dataset = Cicids.Cicids2017(dataframe)
print("Done!")

input_size = dataset.x.shape[1]

output_size = len(labels)

batch_size = 512

#-----Train, Validation and Test DataLoaders-----#
print("Creating train, validation and test dataloaders...")
x_main, x_test, y_main, y_test = utils.splitDataset(dataset.x, dataset.y, 0.7, 0.3)
x_train, x_validation, y_train, y_validation = utils.splitDataset(x_main, y_main, 0.8, 0.2)

train = TensorDataset(x_train, y_train)
validation = TensorDataset(x_validation, y_validation)
test = TensorDataset(x_test, y_test)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
print("Done!")
#-----Train, Validation and Test DataLoaders-----#

MC_model = PredictiveNN(input_size, output_size, device)

VAE_model = VAENN(32, input_size, device)

MC_optimizer = torch.optim.Adam(MC_model.parameters(), lr=0.00001)
VAE_optimizer = torch.optim.Adam(VAE_model.parameters(), lr=0.00001)

MC_criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))

epochs = 150

#-----MultiClass model training-----#
print("Starting MultiClass model Training...")
start = time.time()
MC_model.fit(epochs, MC_optimizer, MC_criterion, train_loader, validation_loader)
MC_model.load_state_dict(torch.load("best_accuracy_scoring_predictive_nn.pt"))
#-----MultiClass model training-----#

#-----VAE model training-----#
print("Starting VirtualAutoEncoder model Training...")
VAE_model.fit(epochs, VAE_optimizer, train_loader, validation_loader)
#-----VAE model training-----#

CPVAE_model = ConcatenatedPredictiveVAE(MC_model, VAE_model, input_size + output_size, output_size, device)

CPVAE_criterion = torch.nn.CrossEntropyLoss()

CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.00001)

#-----CPVAE model training-----#
print("Starting ConcatenatedPredictiveVAE model Training...")
CPVAE_model.fit(epochs, CPVAE_optimizer, CPVAE_criterion, train_loader, validation_loader)
print("Starting Testing...")
accuracy, precision, recall, f1 = CPVAE_model.evaluate(test_loader, CPVAE_criterion)
end = time.time()
print("Test results:")
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
print(f"Tempo di esecuzione: {end - start:.2f} secondi")
#-----CPVAE model training-----#