import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, input_size, conditioning_size):
        super(FiLM, self).__init__()
        self.linear_gamma = nn.Linear(conditioning_size, input_size)
        self.linear_beta = nn.Linear(conditioning_size, input_size)

    def forward(self, x, conditioning, film_switch=True):
        if film_switch:
            gamma = self.linear_gamma(conditioning)
            beta = self.linear_beta(conditioning)
            if len(x.shape) == 4:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            return x * gamma.unsqueeze(1) + beta.unsqueeze(1)
        else:
            return x