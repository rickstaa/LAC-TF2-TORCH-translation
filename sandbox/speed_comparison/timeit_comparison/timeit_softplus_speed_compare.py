"""Script used to check whether the TF and Pytorch softplus method have different
speeds."""

import timeit

# script settings
N_TIMES = int(1e5)
print("Test softplus performance speed in Pytorch and Tensorflow...")

# Pytorch version
setup_str = """
import torch
import numpy as np
import torch.nn.functional as F
a_input_dummy = torch.rand((256,3))
"""
exec_str = """
F.softplus(-2.0 * a_input_dummy)
"""
pytorch_time = timeit.timeit(exec_str, setup=setup_str, number=N_TIMES)

# Tensorflow version
setup_2_str = """
import tensorflow as tf
a_input_dummy = tf.random.uniform((256,3))
@tf.function
def sample_function(a_input_dummy):
    tf.nn.softplus(-2.0 * a_input_dummy)
"""
exec_2_str = """
sample_function(a_input_dummy)
"""
tf_time = timeit.timeit(exec_2_str, setup=setup_2_str, number=N_TIMES)

# Print result
print("\Compare softplus Pytorch/tensorflow speed:")
print(f"- Pytorch pass time: {pytorch_time} s")
print(f"- Tensorflow pass time: {tf_time} s")
