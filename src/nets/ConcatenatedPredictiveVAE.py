import torch
import torch.nn as nn


class ConcatenatedPredictiveVAE(nn.Module):

    def __init__(self, model1, model2, input_size, output_size, device):
        super(ConcatenatedPredictiveVAE, self).__init__()
        self.device = device
        self.mode1 = model1
        self.mode2 = model2
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.to(self.device)

    def forward(self, x1, x2):
        x1 = self.mode1(x1)
        x2 = self.mode2(x2)
        x = torch.cat((x1, x2), dim=1)
        logits = self.fully_connected_1(x)
        return logits