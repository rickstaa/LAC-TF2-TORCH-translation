"""Small test script that analysis if there is a speed difference between pytorch and
tensorflow when using rsample/sample on the normal distribution.
"""

import timeit

# Script settings
N_SAMPLE = int(5e5)  # How many times we sample

######################################################
# Test Lyapunov forward action #######################
######################################################
print("====Forward pass comparison Pytorch/Tensorflow====")
print(
    f"Analysing the speed of performing {N_SAMPLE} forward passes through the "
    "networks (pytorch custom GPU setup)..."
)

#######################################
# Pytorch (Ga Network on GPU) #########
#######################################
print("GA on GPU...")

# Time pytroch sample action
pytorch_setup_code = """
import torch
from gaussian_actor_torch_small import SquashedGaussianMLPActor
from lyapunov_critic_torch import MLPLyapunovCritic
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn autotuner
USE_GPU = True
device = (
    torch.device("cuda")
    if (torch.cuda.is_available() and USE_GPU)
    else torch.device("cpu")
)
ga = SquashedGaussianMLPActor(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[64, 64],
    log_std_min=-20,
    log_std_max=2,
).to(device)
lc =  MLPLyapunovCritic(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[128, 128],
).to(device)
bs = torch.rand((256,8), device="cuda")
ba = torch.rand((256,3), device="cuda")
"""
pytorch_sample_code = """
# Perform forward pass
pi_action, pi_action_det, logp_pi = ga(bs)
L = lc(bs, ba)

# Convert result back to cpu
# pi_action_new = pi_action.cpu().detach().numpy()
# pi_action_det_new = pi_action_det.cpu().detach().numpy()
# logp_pi_new = logp_pi.cpu().detach().numpy()
# L_new = L.cpu().detach().numpy()
"""
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

#######################################
# Tensorflow ##########################
#######################################

# Tensorflowsample action
print("Tensorflow...")
tf_setup_code = """
import tensorflow as tf
from gaussian_actor_tf2_small import SquashedGaussianActor
from lyapunov_critic_tf2 import LyapunovCritic
# tf.config.set_visible_devices([], "GPU") # Disable GPU
ga = SquashedGaussianActor(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[64, 64],
    log_std_min=-20,
    log_std_max=2,
)
lc = LyapunovCritic(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[128, 128],
)
bs = tf.random.uniform((256,8))
ba = tf.random.uniform((256,3))
"""
tf_sample_code = """
_, _, _ = ga(bs)
_ = lc([bs, ba])
"""
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)


######################################################
# Print results ######################################
######################################################
print("\nTest tensorflow/pytorch sample method speed:")
print(f"- Pytorch forward pass time: {pytorch_time} s")
print(f"- Tf forward pass time: {tf_time} s")
