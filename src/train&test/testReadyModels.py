import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.datasets import Cicids
from src.support import utils
from src.nets.PredictiveNN import PredictiveNN
from src.nets.VAENN import VAENN
from src.nets.ConcatenatedPredictiveVAENN import ConcatenatedPredictiveVAE

utils.seed_everything(1) #seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

paths = ["C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
         "C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"]

print("Processing dateset...")
#-----DataFrame-----#
dataframes = utils.readPaths(paths)
labels = utils.stringLabels(dataframes)
dataframes = utils.convertStringsMC(dataframes, labels)
dataframe = pd.concat(dataframes)
#-----DataFrame-----#

week_days_lengths = utils.getWeekDaysLengths(paths)

weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataframe[' Label']), y=dataframe[' Label'])
weights_tensor = torch.Tensor(weights)

dataset = Cicids.Cicids2017(dataframe)

week_days_datasets = utils.splitWeekDaysDatasets(week_days_lengths, dataset) #splitting dataset into single days dataset
print("Done!")

input_size = dataset.x.shape[1]

output_size = len(labels)

batch_size = 512

#-----Train, Validation and Test DataLoaders-----#
print("Creating train, validation and test dataloaders...")
trains = []
tests = []
for dataset in week_days_datasets:
        x_train, x_test, y_train, y_test = utils.splitDataset(dataset.x, dataset.y, 0.7, 0.3)
        trains.append(TensorDataset(x_train, y_train))
        tests.append(TensorDataset(x_test, y_test))
train = ConcatDataset(trains)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loaders = []
for test in tests:
    test_loaders.append(DataLoader(test, batch_size=batch_size, shuffle=True))

x_train, y_train = utils.convertDataLoaderToNumpy(train_loader) #numpy for other models
print("Done!")
#-----Train, Validation and Test DataLoaders-----#

#-----Neural Network models-----#
MC_model = PredictiveNN(input_size, output_size, device)

VAE_model = VAENN(32, input_size, device)

CPVAE_model = ConcatenatedPredictiveVAE(MC_model, VAE_model, input_size + output_size, output_size, device)

knn_model = KNeighborsClassifier()

xgb_model = xgb.XGBClassifier()

rf_model = RandomForestClassifier()

nb_model = GaussianNB()

dt_model = DecisionTreeClassifier()
#-----Neural Network models-----#

MC_optimizer = torch.optim.Adam(MC_model.parameters(), lr=0.00001)
VAE_optimizer = torch.optim.Adam(VAE_model.parameters(), lr=0.00001)
CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.00001)

MC_criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))
CPVAE_criterion = torch.nn.CrossEntropyLoss()

epochs = 150

days = ["Tuesday", "Wednesday", "Thursday", "Friday"]
print("Starting concatenatedPredictiveVAE model training...")
start = time.time()
#-----MultiClass model training-----#
MC_model.fit(epochs, MC_optimizer, MC_criterion, train_loader)
#-----MultiClass model training-----#

#-----VAE model training-----#
VAE_model.fit(epochs, VAE_optimizer, train_loader)
#-----VAE model training-----#

#-----CPVAE model training-----#
CPVAE_model.fit(epochs, CPVAE_optimizer, CPVAE_criterion, train_loader)
#-----CPVAE model training-----#
end = time.time()
print("CPVAE done!")
print(f"Training time: {end - start:.2f} seconds")

print("Starting KNN model training...")
knn_start = time.time()
#-----KNN model training-----#
knn_model.fit(x_train, y_train)
#-----KNN model training-----#
knn_end = time.time()
print("KNN done!")
print(f"Training time: {knn_end - knn_start:.2f} seconds")

print("Starting XGBoost model training...")
xgb_start = time.time()
#-----XGBoost model training-----#
xgb_model.fit(x_train, y_train)
#-----XGBoost model training-----#
xgb_end = time.time()
print("XGBoost done!")
print(f"Training time: {xgb_end - xgb_start:.2f} seconds")

print("Starting RandomForest model training...")
rf_start = time.time()
#-----RandomForest model training-----#
rf_model.fit(x_train, y_train)
#-----RandomForest model training-----#
rf_end = time.time()
print("RandomForest done!")
print(f"Training time: {rf_end - rf_start:.2f} seconds")

print("Starting NaiveBayes model training...")
nb_start = time.time()
#-----NaiveBayes model training-----#
nb_model.fit(x_train, y_train)
#-----NaiveBayes model training-----#
nb_end = time.time()
print("NaiveBayes done!")
print(f"Training time: {nb_end - nb_start:.2f} seconds")

print("Starting DecisionTree model training...")
dt_start = time.time()
#-----Decisiontree model training-----#
dt_model.fit(x_train, y_train)
#-----DecisionTree model training-----#
dt_end = time.time()
print("DecisionTree done!")
print(f"Training time: {dt_end - dt_start:.2f} seconds")

for i in range(len(test_loaders)):
        print(f"Starting {days[i]} ConcatenatedPredictiveVAE testing...")
        accuracy, precision, recall, f1 = CPVAE_model.evaluate(test_loaders[i], CPVAE_criterion)
        end = time.time()
        print("ConcatenatedPredictiveVAE test results:")
        print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

        x_test, y_test = utils.convertDataLoaderToNumpy(test_loaders[i]) #numpy for other models
        print(f"Starting {days[i]} knn testing...")
        knn_pred = knn_model.predict(x_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        knn_precision = precision_score(y_test, knn_pred)
        knn_recall = recall_score(y_test, knn_pred)
        knn_f1 = f1_score(y_test, knn_pred)
        print("KNN test results:")
        print(f"accuracy: {knn_accuracy}, precision: {knn_precision}, recall: {knn_recall}, f1: {knn_f1}")

        print(f"Starting {days[i]} XGBoost testing...")
        xgb_pred = xgb_model.predict(x_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_precision = precision_score(y_test, xgb_pred)
        xgb_recall = recall_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred)
        print("XGBoost test results:")
        print(f"accuracy: {xgb_accuracy}, precision: {xgb_precision}, recall: {xgb_recall}, f1: {xgb_f1}")

        print(f"Starting {days[i]} RandomForest testing...")
        rf_pred = xgb_model.predict(x_test)
        rf_accuracy = accuracy_score(y_test, xgb_pred)
        rf_precision = precision_score(y_test, xgb_pred)
        rf_recall = recall_score(y_test, xgb_pred)
        rf_f1 = f1_score(y_test, xgb_pred)
        print("RandomForest test results:")
        print(f"accuracy: {rf_accuracy}, precision: {rf_precision}, recall: {rf_recall}, f1: {rf_f1}")

        print(f"Starting {days[i]} NaiveBayes testing...")
        nb_pred = xgb_model.predict(x_test)
        nb_accuracy = accuracy_score(y_test, xgb_pred)
        nb_precision = precision_score(y_test, xgb_pred)
        nb_recall = recall_score(y_test, xgb_pred)
        nb_f1 = f1_score(y_test, xgb_pred)
        print("NaiveBayes test results:")
        print(f"accuracy: {nb_accuracy}, precision: {nb_precision}, recall: {nb_recall}, f1: {nb_f1}")

        print(f"Starting {days[i]} DecisionTree testing...")
        dt_pred = xgb_model.predict(x_test)
        dt_accuracy = accuracy_score(y_test, xgb_pred)
        dt_precision = precision_score(y_test, xgb_pred)
        dt_recall = recall_score(y_test, xgb_pred)
        dt_f1 = f1_score(y_test, xgb_pred)
        print("DecisionTree test results:")
        print(f"accuracy: {dt_accuracy}, precision: {dt_precision}, recall: {dt_recall}, f1: {dt_f1}")