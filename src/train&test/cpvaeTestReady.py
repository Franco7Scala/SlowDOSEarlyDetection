import torch
import time
import pickle
import os.path

from src.nets.PredictiveExtendedNN import PredictiveExtendedNN
from src.support import utils
from src.nets.PredictiveNN import PredictiveNN
from src.nets.VAENN import VAENN
from src.nets.ConcatenatedPredictiveVAENN import ConcatenatedPredictiveVAE
from src.support.utils import get_base_dir

utils.seed_everything(1) #seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = pickle.load(open(f"{get_base_dir()}/files/input_size.pkl", 'rb'))
output_size = 2
ddos_weights = pickle.load(open(f"{get_base_dir()}/files/ddos_weights.pkl", 'rb'))
slowdos_weights = pickle.load(open(f"{get_base_dir()}/files/slowdos_weights.pkl", 'rb'))
ddos_train_loader = pickle.load(open(f"{get_base_dir()}/files/ddos_train_loader.pkl", 'rb'))
slowdos_train_loader = pickle.load(open(f"{get_base_dir()}/files/slowdos_train_loader.pkl", 'rb'))
slowdos_test_loader = pickle.load(open(f"{get_base_dir()}/files/slowdos_test_loader.pkl", 'rb'))
vae_path = f"{get_base_dir()}/files/vae_model_extended.pt"

epochs_ff_first_train = 150
epochs_vae_first_train = 150
epochs_ensemble_first_train = 50
epochs_ensemble_adaptation_train = 150
dim_code = 8
dropout = 0.05


#-----ConcatenatedPredictiveVAE NeuralNetwork-----#
MC_model = PredictiveNN(input_size, output_size, device, dropout)
MC_model_extended = PredictiveExtendedNN(input_size, output_size, device, dropout)
VAE_model = VAENN(dim_code, input_size, device)

CPVAE_model = ConcatenatedPredictiveVAE(MC_model, MC_model_extended, VAE_model, (dim_code + 32), output_size, device)

MC_optimizer = torch.optim.Adam(MC_model.parameters(), lr=0.0001)
VAE_optimizer = torch.optim.Adam(VAE_model.parameters(), lr=0.0001)
CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.0001)

MC_criterion = torch.nn.CrossEntropyLoss(weight=ddos_weights.to(device))
CPVAE_criterion = torch.nn.CrossEntropyLoss(weight=ddos_weights.to(device))
#-----ConcatenatedPredictiveVAE NeuralNetwork-----#

print("Starting DDoS ConcatenatedPredictiveVAE model training...")
start = time.time()
#-----MultiClass model training-----#
MC_model.fit(epochs_ff_first_train, MC_optimizer, MC_criterion, ddos_train_loader)
#-----MultiClass model training-----#

#-----VAE model training-----#
if os.path.isfile(vae_path):
    print("Loading VAE model...")
    torch.load(VAE_model, vae_path)

else:
    print("Training VAE model...")
    VAE_model.fit(epochs_vae_first_train, VAE_optimizer, ddos_train_loader)
    torch.save(VAE_model, vae_path)
    print("VAE model saved!")
#-----VAE model training-----#

#-----CPVAE model training-----#
CPVAE_model.fit(epochs_ensemble_first_train, CPVAE_optimizer, CPVAE_criterion, ddos_train_loader)
#-----CPVAE model training-----#

end = time.time()

print(f"Training time base model: {end - start:.2f} seconds")

start = time.time()

for param in MC_model.parameters():
    param.requires_grad = True


#MC_criterion = torch.nn.CrossEntropyLoss(weight=slowdos_weights.to(device))
CPVAE_criterion = torch.nn.CrossEntropyLoss(weight=slowdos_weights.to(device))

print("Starting SlowDoS ConcatenatedPredictiveVAE model training...")
#-----MultiClass model training-----#
#MC_model.fit(epochs, MC_optimizer, MC_criterion, slowdos_train_loader)
#-----MultiClass model training-----#

#-----CPVAE model training-----#
CPVAE_model.fit(epochs_ensemble_adaptation_train, CPVAE_optimizer, CPVAE_criterion, slowdos_train_loader)
#-----CPVAE model training-----#
end = time.time()

print("ConcatenatedPredictiveVAE done!")
print(f"Adaptation training time: {end - start:.2f} seconds")

print(f"Starting ConcatenatedPredictiveVAE testing...")
accuracy, precision, recall, f1, cr = CPVAE_model.evaluate(slowdos_test_loader, CPVAE_criterion)
print("ConcatenatedPredictiveVAE test results:")
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
print(cr)