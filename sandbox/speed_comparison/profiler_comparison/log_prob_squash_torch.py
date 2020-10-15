"""Small test script that analysis if there is a speed difference between Pytorch and
Tensorflow of performing a squashing operation to a action and a distribution.
"""

import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Script settings
N_SAMPLE = int(5e5)  # How many times we sample
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.be4nchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
batch_size = 256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)


# @torch.jit.script
# def squash_correct(pi_action):
#     return 2 * (
#         torch.log(torch.tensor(2)) - pi_action - F.softplus(-2 * pi_action)
#     ).sum(axis=1)


# Perform log_prob calculation and squashing in a loop
print(f"Performing Torch log_prob squash operation inside a loop {N_SAMPLE} times")
start_time = time.time()
for ii in range(N_SAMPLE):
    pi_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
    pi_action = torch.rand(batch_size, 3)
    pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1
    logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
    # logp_pi -= squash_correct(pi_action)
    logp_pi -= (
        2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))
    ).sum(axis=1)
duration = time.time() - start_time

# Print duration
print(f"The duration was: {duration} s")
