import torch
import numpy as np

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def get_distribution_input_sampler():
    return lambda m, n: torch.rand(m, n) 
