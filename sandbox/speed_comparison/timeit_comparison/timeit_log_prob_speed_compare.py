"""Small test script that analysis if there is a speed difference between pytorch and
tensorflow when using rsample/sample on the normal distribution.
"""

import timeit

# Script settings
N_SAMPLE = int(1e6)  # How many times we sample

######################################################
# Test logprob calculation ###########################
######################################################
print("====Log_prob speed comparison Pytorch/Tensorflow====")
print(
    "Analysing the speed of calculating the log_prob of an action that is "
    f"r-sampled from the normal distribution {N_SAMPLE} times..."
)

# Time pytroch sample action
pytorch_setup_code = """
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
batch_size=256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)
"""
pytorch_sample_code = """
pi_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
pi_action = (
    pi_distribution.rsample()
)  # Sample while using the parameterization trick
logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
"""
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Tensorflowsample action
tf_setup_code = """
import tensorflow as tf
import tensorflow_probability as tfp
batch_size=256
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)
@tf.function
def sample_function():
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon = base_distribution.sample(batch_size)
    raw_action = affine_bijector.forward(epsilon)
    log_pi = base_distribution.log_prob(raw_action)
"""
tf_sample_code = """
sample_function()
"""
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

######################################################
# Test logprob calculation + squash ##################
######################################################
print(
    "Analysing the speed of calculating the log_prob of an action that is "
    f"r-sampled from the normal distribution {N_SAMPLE} times. While also squashing "
    "the output..."
)

# Time pytroch sample action
pytorch_setup_code = """
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
batch_size=256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)
"""
pytorch_sample_code = """
pi_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
pi_action = (
    pi_distribution.rsample()
)  # Sample while using the parameterization trick
pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1
logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
    axis=1
)
"""
pytorch_time_2 = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Tensorflowsample action
tf_setup_code = """
import tensorflow as tf
import tensorflow_probability as tfp
from squash_bijector import SquashBijector
batch_size=256
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)
@tf.function
def sample_function():
    squash_bijector = SquashBijector()
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon = base_distribution.sample(batch_size)
    raw_action = affine_bijector.forward(epsilon)
    clipped_a = squash_bijector.forward(raw_action)
    reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
    distribution = tfp.distributions.TransformedDistribution(
        distribution=base_distribution, bijector=reparm_trick_bijector
    )
    logp_pi = distribution.log_prob(clipped_a)
"""
tf_sample_code = """
sample_function()
"""
tf_time_2 = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

######################################################
# Print results ######################################
######################################################
print("\nTest tensorflow/pytorch log_prob method speed:")
print(f"- Pytorch log_prob time: {pytorch_time} s")
print(f"- Tf log_prob time: {tf_time} s")
print("\nTest tensorflow/pytorch log_prob + squash method speed:")
print(f"- Pytorch log_prob squash time: {pytorch_time_2} s")
print(f"- Tf log_prob squash time: {tf_time_2} s")
