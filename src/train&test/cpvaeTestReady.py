import torch
import time
import pickle

from src.support import utils
from src.nets.PredictiveNN import PredictiveNN
from src.nets.VAENN import VAENN
from src.nets.ConcatenatedPredictiveVAENN import ConcatenatedPredictiveVAE

utils.seed_everything(1) #seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/input_size.pkl', 'rb'))
output_size = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/output_size.pkl', 'rb'))
weights = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/weights_tensor.pkl', 'rb'))
train_loader = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/train_loader.pkl', 'rb'))
test_loaders = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/test_loaders.pkl', 'rb'))

epochs = 150

days = ["Tuesday", "Wednesday", "Thursday", "Friday"]

#-----ConcatenatedPredictiveVAE NeuralNetwork-----#
MC_model = PredictiveNN(input_size, output_size, device)
VAE_model = VAENN(32, input_size, device)
CPVAE_model = ConcatenatedPredictiveVAE(MC_model, VAE_model, input_size + output_size, output_size, device)

MC_optimizer = torch.optim.Adam(MC_model.parameters(), lr=0.00001)
VAE_optimizer = torch.optim.Adam(VAE_model.parameters(), lr=0.00001)
CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.00001)

MC_criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
CPVAE_criterion = torch.nn.CrossEntropyLoss()
#-----ConcatenatedPredictiveVAE NeuralNetwork-----#

print("Starting ConcatenatedPredictiveVAE model training...")
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
print("ConcatenatedPredictiveVAE done!")
print(f"Training time: {end - start:.2f} seconds")

for i in range(len(test_loaders)):
        print(f"Starting {days[i]} ConcatenatedPredictiveVAE testing...")
        accuracy, precision, recall, f1 = CPVAE_model.evaluate(test_loaders[i], CPVAE_criterion)
        end = time.time()
        print("ConcatenatedPredictiveVAE test results:")
        print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")