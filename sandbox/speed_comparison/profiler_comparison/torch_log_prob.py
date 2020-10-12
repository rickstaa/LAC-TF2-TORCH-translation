"""Pytorch log_probability calculation version. Used to
compare which version is faster.
"""

import time

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Script settings
N_SAMPLE = int(1e3)  # How many times we sample
batch_size = 256

# Create dummy inputs
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)

# Calculate log probabilities in loop
print(f"Use Pytorch to compute the log_prob for dummy arrays {N_SAMPLE} time.")
start_time = time.time()
for _ in range(0, N_SAMPLE):
    pi_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
    pi_action = (
        pi_distribution.rsample()
    )  # Sample while using the parameterization trick
    logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
duration = time.time() - start_time
print(f"Duration: {duration} s")
