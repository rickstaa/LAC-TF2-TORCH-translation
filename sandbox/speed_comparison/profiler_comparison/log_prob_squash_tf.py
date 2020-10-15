"""Small test script that analysis if there is a speed difference between Pytorch and
Tensorflow of performing a squashing operation to a action and a distribution.
"""

import time
import tensorflow as tf
import tensorflow_probability as tfp
from squash_bijector import SquashBijector

# Script settings
N_SAMPLE = int(5e5)  # How many times we sample
# tf.config.set_visible_devices([], "GPU") # Disable GPU
batch_size = 256
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)


@tf.function
def sample_function():
    squash_bijector = SquashBijector()
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon_dummy = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    pi_action = affine_bijector.forward(epsilon_dummy)
    pi_action = squash_bijector.forward(pi_action)
    reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
    distribution = tfp.distributions.TransformedDistribution(
        distribution=base_distribution, bijector=reparm_trick_bijector
    )
    logp_pi = distribution.log_prob(pi_action)


# Perform calculation inside loop
print(f"Performing TF log_prob squash operation inside a loop {N_SAMPLE} times")
start_time = time.time()
for ii in range(N_SAMPLE):
    sample_function()
duration = time.time() - start_time

# Print duration
print(f"The duration was: {duration} s")
