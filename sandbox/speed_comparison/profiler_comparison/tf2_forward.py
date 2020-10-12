"""Tensorflow 2.0 forward version. Used to
compare which version is faster.
"""

import time

import tensorflow as tf
from gaussian_actor_tf2 import SquashedGaussianActor
from lyapunov_critic_tf2 import LyapunovCritic

# SCript settings
N_SAMPLE = int(1e3)  # How many times we sample

# Create networks
ga = SquashedGaussianActor(
    obs_dim=8, act_dim=3, hidden_sizes=[64, 64], log_std_min=-20, log_std_max=2,
)
lc = LyapunovCritic(obs_dim=8, act_dim=3, hidden_sizes=[128, 128],)

# Create dummy variables
bs = tf.random.uniform((256, 8))
ba = tf.random.uniform((256, 3))

# Perform forward pass in loop
print(f"Use tf2 to perform {N_SAMPLE} forward passes through the network.")
start_time = time.time()
for _ in range(0, N_SAMPLE):
    _, _, _ = ga(bs)
    _ = lc([bs, ba])
duration = time.time() - start_time
print(f"Duration: {duration} s")
