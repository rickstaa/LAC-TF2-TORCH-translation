"""Pytorch forward version. Used to
compare which version is faster.
"""

import time

import torch
from gaussian_actor_torch import SquashedGaussianMLPActor
from lyapunov_critic_torch import MLPLyapunovCritic

# SCript settings
N_SAMPLE = int(1e3)  # How many times we sample

# Create networks
ga = SquashedGaussianMLPActor(
    obs_dim=8, act_dim=3, hidden_sizes=[64, 64], log_std_min=-20, log_std_max=2,
)
lc = MLPLyapunovCritic(obs_dim=8, act_dim=3, hidden_sizes=[128, 128],)

# Create dummy variables
bs = torch.rand((256, 8))
ba = torch.rand((256, 3))

# Perform forward pass in loop
print(f"Use Pytorch to perform {N_SAMPLE} forward passes through the network.")
start_time = time.time()
for _ in range(0, N_SAMPLE):
    _, _, _ = ga(bs)
    _ = lc(bs, ba)
duration = time.time() - start_time
print(f"Duration: {duration} s")
