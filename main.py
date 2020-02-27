import torch
import numpy as np

def get_distribution_sampler(mu, sigma):
  return lambda x: torch.Tensor(np.random.normal(mu, sigma, (1, n)))
