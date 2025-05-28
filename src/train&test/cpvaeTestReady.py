import torch
import time
import pickle
import os.path

from src.nets.PredictiveExtendedNN import PredictiveExtendedNN
from src.support import utils
from src.nets.PredictiveNN import PredictiveNN
from src.nets.VAENN import VAENN
from src.nets.ConcatenatedPredictiveVAENN import ConcatenatedPredictiveVAE
from src.support.focal_loss import FocalLoss
from src.support.utils import get_base_dir

utils.seed_everything(1) #seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

input_size = pickle.load(open(f"{get_base_dir()}/pickels/input_size.pkl", 'rb'))
output_size = 2
first_weights = pickle.load(open(f"{get_base_dir()}/pickels/dos_weights.pkl", 'rb'))
adaptation_weights = pickle.load(open(f"{get_base_dir()}/pickels/slowdos_weights.pkl", 'rb'))
first_train_loader = pickle.load(open(f"{get_base_dir()}/pickels/dos_train_loader.pkl", 'rb'))
adaptation_train_loader = pickle.load(open(f"{get_base_dir()}/pickels/slowdos_train_loader.pkl", 'rb'))
slowdos_test_loader = pickle.load(open(f"{get_base_dir()}/pickels/slowdos_test_loader.pkl", 'rb'))
vae_path = f"{get_base_dir()}/vae_model_extended.pt"

epochs_ff_first_train = 150
epochs_vae_first_train = 150
epochs_ensemble_first_train = 50
epochs_ensemble_adaptation_train = 150
dim_code = 8
dropout = 0.05
weighs = [-1, 32]
random_noise = True
mean = 0.0
std = 0.05


#-----ConcatenatedPredictiveVAE NeuralNetwork-----#
MC_model = PredictiveNN(input_size, output_size, device, dropout)
MC_model_extended = PredictiveExtendedNN(input_size, output_size, device, dropout)
VAE_model = VAENN(dim_code, input_size, device)

CPVAE_model = ConcatenatedPredictiveVAE(MC_model, MC_model_extended, VAE_model, (dim_code + 32), output_size, device, random_noise=random_noise, mean=mean, std=std)

MC_optimizer = torch.optim.Adam(MC_model.parameters(), lr=0.0001)
VAE_optimizer = torch.optim.Adam(VAE_model.parameters(), lr=0.0001)
CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.0001)

MC_criterion = torch.nn.CrossEntropyLoss(weight=first_weights.to(device))
CPVAE_criterion = torch.nn.CrossEntropyLoss(weight=first_weights.to(device))
#-----ConcatenatedPredictiveVAE NeuralNetwork-----#

print("Starting DDoS ConcatenatedPredictiveVAE model training...")
start = time.time()
#-----MultiClass model training-----#
MC_model.fit(epochs_ff_first_train, MC_optimizer, MC_criterion, first_train_loader)
#-----MultiClass model training-----#

#-----VAE model training-----#
if os.path.isfile(vae_path):
    print("Loading VAE model...")
    VAE_model.load_state_dict(torch.load(vae_path, weights_only=True))

else:
    print("Training VAE model...")
    VAE_model.fit(epochs_vae_first_train, VAE_optimizer, first_train_loader)
    torch.save(VAE_model.state_dict(), vae_path)
    print("VAE model saved!")
#-----VAE model training-----#

#-----CPVAE model training-----#
CPVAE_model.fit(epochs_ensemble_first_train, CPVAE_optimizer, CPVAE_criterion, first_train_loader)
#-----CPVAE model training-----#

end = time.time()

print(f"Training time base model: {end - start:.2f} seconds")

start = time.time()

for param in MC_model.parameters():
    param.requires_grad = True


#MC_criterion = torch.nn.CrossEntropyLoss(weight=slowdos_weights.to(device))
if weighs is not None:
    if weighs[0] != -1:
        adaptation_weights[0] = weighs[0]

    if weighs[1] != -1:
        adaptation_weights[1] = weighs[1]

#CPVAE_criterion = torch.nn.CrossEntropyLoss(weight=slowdos_weights.to(device))
CPVAE_criterion = FocalLoss(gamma=2, alpha=0.5, reduction="mean")

print("Starting SlowDoS ConcatenatedPredictiveVAE model training...")
#-----MultiClass model training-----#
#MC_model.fit(epochs, MC_optimizer, MC_criterion, slowdos_train_loader)
#-----MultiClass model training-----#

#-----CPVAE model training-----#
CPVAE_model.fit(epochs_ensemble_adaptation_train, CPVAE_optimizer, CPVAE_criterion, adaptation_train_loader)
#-----CPVAE model training-----#
end = time.time()

print("ConcatenatedPredictiveVAE done!")
print(f"Adaptation training time: {end - start:.2f} seconds")

print(f"Starting ConcatenatedPredictiveVAE testing on train set...")
accuracy, precision, recall, f1, auc, cr = CPVAE_model.evaluate(adaptation_train_loader, CPVAE_criterion, evaluation_on="train")
print("ConcatenatedPredictiveVAE test results:")
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}")
print(cr)

print("-" * 100)

print(f"Starting ConcatenatedPredictiveVAE testing on test set...")
accuracy, precision, recall, f1, auc, cr = CPVAE_model.evaluate(slowdos_test_loader, CPVAE_criterion, evaluation_on="test")
print("ConcatenatedPredictiveVAE test results:")
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}")
print(cr)