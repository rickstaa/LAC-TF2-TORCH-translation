"""Small test script that analysis if there is a speed difference between Pytorch and
Tensorflow when computing the log_prob of a dummy action.
"""

import timeit

# Script settings
N_SAMPLE = int(5e5)  # How many times we sample

######################################################
# Test logprob calculation ###########################
######################################################
print("====Log_prob speed comparison Pytorch/Tensorflow====")
print(
    "Analysing the speed of calculating the log_prob of a dummy action on the normal "
    f"distribution for {N_SAMPLE} times..."
)

# Time pytroch sample action
pytorch_setup_code = """
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
batch_size=256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)
"""
pytorch_sample_code = """
pi_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
pi_action_dummy = torch.rand(batch_size,3)
logp_pi = pi_distribution.log_prob(pi_action_dummy).sum(axis=-1)
"""
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Tensorflowsample action
tf_setup_code = """
import tensorflow as tf
import tensorflow_probability as tfp
tf.config.set_visible_devices([], "GPU") # Disable GPU
batch_size=256
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)
@tf.function
def sample_function():
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon_dummy = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    raw_action = affine_bijector.forward(epsilon_dummy)
    log_pi = base_distribution.log_prob(raw_action)
"""
tf_sample_code = """
sample_function()
"""
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

######################################################
# Print results ######################################
######################################################
print("\nTest tensorflow/pytorch log_prob method speed:")
print(f"- Pytorch log_prob time: {pytorch_time} s")
print(f"- Tf log_prob time: {tf_time} s")
