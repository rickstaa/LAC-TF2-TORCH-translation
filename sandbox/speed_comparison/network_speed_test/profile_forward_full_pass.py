"""Small test script that profile the speed of a forward pass through the networks.
"""

import time
import torch
from gaussian_actor_torch import SquashedGaussianMLPActor
from lyapunov_critic_torch import MLPLyapunovCritic

# Script settings
N_SAMPLE = int(1e4)  # How many times we sample

######################################################
# Test Lyapunov forward action #######################
######################################################
print("====Forward pass comparison Pytorch/Tensorflow====")
print(
    f"Analysing the speed of performing {N_SAMPLE} forward passes through the "
    "networks..."
)

# GPU settings
# torch.set_default_tensor_type("torch.cuda.FloatTensor")  # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner  # Enable global GPU speedup tricks level 1
# torch.backends.cudnn.fastest = (
#     True  # Enable cudnn fastest autotuner # Enable global GPU speedup tricks level 2
# )

# Create inputs
bs = torch.rand((256, 8))
ba = torch.rand((256, 3))

# Create networks
ga = SquashedGaussianMLPActor(
    obs_dim=8, act_dim=3, hidden_sizes=[64, 64], log_std_min=-20, log_std_max=2,
)
lc = MLPLyapunovCritic(obs_dim=8, act_dim=3, hidden_sizes=[128, 128],)

# Perform forward pass for N_Times
start_time = time.time()
for ii in range(0, N_SAMPLE):
    _, _, _ = ga(bs)
    _ = lc(bs, ba)
total_time = time.time() - start_time

######################################################
# Print results ######################################
######################################################
print("\nTest tensorflow/pytorch sample method speed:")
print(f"- Pytorch total forward pass time: {total_time} s")
print("- Pytorch mean forward pass time: {} ms".format((total_time / N_SAMPLE) * 1e3))
