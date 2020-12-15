"""Small test script that analysis if there is a speed difference between pytorch and
tensorflow when using rsample/sample on the normal distribution.
"""

import timeit

# Script settings
N_SAMPLE = int(1e5)  # How many times we sample

######################################################
# Test Lyapunov forward action #######################
######################################################
print("====Forward pass comparison Pytorch/Tensorflow====")
print(
    f"Analysing the speed of performing {N_SAMPLE} forward passes through the "
    "networks..."
)

# # Time pytroch sample action
# pytorch_setup_code = """
# import torch
# from gaussian_actor_torch import SquashedGaussianMLPActor
# from lyapunov_critic_torch import MLPLyapunovCritic
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# # torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
# ga = SquashedGaussianMLPActor(
#     obs_dim=8,
#     act_dim=3,
#     hidden_sizes=[64, 64],
#     log_std_min=-20,
#     log_std_max=2,
# )
# lc =  MLPLyapunovCritic(
#     obs_dim=8,
#     act_dim=3,
#     hidden_sizes=[128, 128],
# )
# bs = torch.rand((256,8))
# ba = torch.rand((256,3))
# """
# pytorch_sample_code = """
# _, _, _ = ga(bs)
# _ = lc(bs, ba)
# """
# pytorch_time = timeit.timeit(
#     pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
# )

# Tensorflowsample action
tf_setup_code = """
import tensorflow as tf
from gaussian_actor_tf2 import SquashedGaussianActor
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
# print(f"- Pytorch forward pass time: {pytorch_time} s")
# print("- Pytorch mean forward pass time: {} ms".format((pytorch_time / N_SAMPLE) * 1e3))
print(f"- Total Tf forward pass time: {tf_time} s")
print(f"- Tensorflow total forward pass time: {tf_time} s")
print("- Tensorflow mean forward pass time: {} ms".format((tf_time / N_SAMPLE) * 1e3))
