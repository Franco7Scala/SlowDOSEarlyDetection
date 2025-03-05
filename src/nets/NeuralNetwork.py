import torch
import torch.nn as nn
from torch.nn import Softmax


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, 55),
            nn.ReLU(inplace=True),
            nn.Linear(55, 39),
            nn.ReLU(inplace=True),
            nn.Linear(39, output_size),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = x.to(self.device).float()
        logits = self.fully_connected_1(x)
        pred_probab = Softmax(dim=1)(logits)
        return pred_probab