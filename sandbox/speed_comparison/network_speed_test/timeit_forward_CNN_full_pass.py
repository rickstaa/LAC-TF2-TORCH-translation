"""Small test script that analysis the speed of a forward pass through the networks./
"""

import timeit

# Script settings
N_SAMPLE = int(1e2)  # How many times we sample

######################################################
# Test Lyapunov forward action #######################
######################################################
print("====Forward pass comparison Pytorch/Tensorflow====")
print(
    f"Analysing the speed of performing {N_SAMPLE} forward passes through the "
    "networks..."
)

# Time pytorch sample action
pytorch_setup_code = """
import numpy as np
import torch
from gaussian_actor_torch_cnn import SquashedGaussianMLPActor
from lyapunov_critic_torch_cnn import MLPLyapunovCritic

# GPU settings
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner  # Enable global GPU speedup tricks level 1
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner # Enable global GPU speedup tricks level 2

# Settings
BATCH_SIZE = 256
IMG_SIZE = [3, 128, 128]
ACTOR_SIZE = [64, 64]
CRITIC_SIZE = [128, 128]
CNN_OUTPUT_SIZE = 10  # The output size of the cnn before flattening it

# Create input image
bs = torch.tensor(
    np.random.randint(255, size=(BATCH_SIZE, *IMG_SIZE), dtype=np.uint8),
    dtype=torch.float32,
)
ba = torch.rand((BATCH_SIZE, 3))

# Create networks
ga = SquashedGaussianMLPActor(
    obs_dim=IMG_SIZE,
    act_dim=3,
    hidden_sizes=ACTOR_SIZE,
    log_std_min=-20,
    log_std_max=2,
    cnn_output_size=CNN_OUTPUT_SIZE,
)
lc = MLPLyapunovCritic(
    obs_dim=IMG_SIZE,
    act_dim=3,
    hidden_sizes=CRITIC_SIZE,
    cnn_output_size=CNN_OUTPUT_SIZE,
)
"""
pytorch_sample_code = """
_, _, _ = ga(bs)
_ = lc(bs, ba)
"""
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

######################################################
# Print results ######################################
######################################################
print("\nTest tensorflow/pytorch sample method speed:")
print(f"- Pytorch total forward pass time: {pytorch_time} s")
print("- Pytorch mean forward pass time: {} ms".format((pytorch_time / N_SAMPLE) * 1e3))
