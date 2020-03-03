import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

###### DATA. Parameters and functions regarding the data

# Data params
data_mean = 4
data_stddev = 1.25

# Target data and generator input data functions
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n))) # Gaussian

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n) # Uniform distribution, not Gaussian

###### MODELS.

# Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        x = nn.tanh(x)
        x = self.linear3(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.sigmoid(x)
        x = self.linear2(x)
        x = nn.sigmoid(x)
        x = self.linear3(x)
        x = nn.sigmoid(x)
        return x

###### TRAIN.

def train():
    # Model parameters
    g_input_size = 1 # 
    g_hidden_size = 5

    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            # 1. Train D on real data + fake data
            D.zero_grad()

            # 1A. Train D on real data
            #d_real_data = Variable(d_sampler(d_input_size)) # generate real data
            #d_real_decision = D(preprocess(d_real_data)) # forward pass it in the discriminator
            #d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1]))) # ones = true
            #d_real_error.backward() # compute/store gradients, but don't change params
