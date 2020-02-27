import torch
import numpy as np
import torch.nn as nn

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def get_distribution_input_sampler():
    return lambda m, n: torch.rand(m, n)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.linear1(x)
        x = self.f(x)
        x = self.linear2(x)
        x = self.f(x)
        x = self.linear3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.linear1(x)
        x = self.f(x)
        x = self.linear2(x)
        x = self.f(x)
        x = self.linear3(x)
        x = self.f(x)
        return x
