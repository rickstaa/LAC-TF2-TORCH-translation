"""Tensorflow 2.0 sample version. Used to
compare which version is faster.
"""

import time

import tensorflow as tf
import tensorflow_probability as tfp

# SCript settings
N_SAMPLE = int(1e3)  # How many times we sample
batch_size = 256

# Create dummy arrays
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)


# Create tf function
@tf.function
def sample_function():
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    normal_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon = normal_distribution.sample(batch_size)
    _ = affine_bijector.forward(epsilon)


# Rsample in loop
print(f"Use Pytorch to compute the log_prob for dummy arrays {N_SAMPLE} time.")
start_time = time.time()
for _ in range(0, N_SAMPLE):
    sample_function()
duration = time.time() - start_time
print(f"Duration: {duration} s")
