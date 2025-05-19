import torch
import math
from torch import nn


class PredictiveExtendedNN(nn.Module):
    def __init__(self, input_size, output_size, device, dropout: float):
        super(PredictiveExtendedNN, self).__init__()
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return create_extended_input(x)


def create_extended_input(raw_input_layer):
    extended_features = []
    range_values = [8]

    #extended_features.append(raw_input_layer)

    #one_minus_i = Lambda(lambda x: 1 - torch.clip(x, 0, 1))(raw_input_layer)
    #extended_features.append(one_minus_i)

    # power
    for v in range_values:
        power_i = LambdaLayer(lambda x: x ** v)(raw_input_layer)
        extended_features.append(power_i)

    # root
    for v in range_values:
        root_i = LambdaLayer(lambda x: torch.clip(x, 0, 1) ** (1 / v))(raw_input_layer)
        extended_features.append(root_i)

    #sin and 1-cos
    #sin_i = Lambda(lambda x: torch.sin(math.pi * torch.clip(x, 0, 1)))(raw_input_layer)
    #extended_features.append(sin_i)
    #one_minus_cos_i = Lambda(lambda x: 1 - torch.cos(math.pi * torch.clip(x, 0, 1)))(raw_input_layer)
    #extended_features.append(one_minus_cos_i)

    # other extensions
    log_i = LambdaLayer(lambda x: torch.log(torch.clip(x, 0, 1) + 1) / math.log(2))(raw_input_layer)
    extended_features.append(log_i)
    #one_minus_inv_log_i = Lambda(lambda x: 1 - torch.log(torch.clip(-x, 0, 1) + 2) / math.log(2))(raw_input_layer)
    #extended_features.append(one_minus_inv_log_i)
    exp_i = LambdaLayer(lambda x: torch.exp(x - 1))(raw_input_layer)
    extended_features.append(exp_i)
    #one_minus_exp_i = Lambda(lambda x: 1 - torch.exp(-x))(raw_input_layer)
    #extended_features.append(one_minus_exp_i)

    # improved input
    return torch.cat(extended_features, axis=1)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
