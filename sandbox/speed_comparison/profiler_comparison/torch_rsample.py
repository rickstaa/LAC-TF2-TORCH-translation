"""Pytorch sample version. Used to
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

# Create dummy arrays
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)

# Rsample in loop
print(f"Use Pytorch to compute the log_prob for dummy arrays {N_SAMPLE} time.")
start_time = time.time()
for _ in range(0, N_SAMPLE):
    normal_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
    pi_action = (
        normal_distribution.rsample()
    )  # Sample while using the parameterization trick
duration = time.time() - start_time
print(f"Duration: {duration} s")
