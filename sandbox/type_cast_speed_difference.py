"""Script to investigate the fastest way to convert a float62 numpy array to a
torch.float32 tensor.
"""
import numpy as np
import tensorflow as tf
import torch

# # == USING timeit ==
# import timeit

# # Time numpy casting
# np_cast_str = """
# import numpy as np
# arr_64 = np.random.rand(3, 2)
# test = arr_64.astype(np.float32)
# """
# np_cast_time = timeit.timeit(np_cast_str, number=1000)

# # Time tensorflow casting
# tf_cast_str = """
# import numpy as np
# import tensorflow as tf
# arr_64 = np.random.rand(3, 2)
# test = tf.convert_to_tensor(arr_64, dtype=tf.float32)
# """
# tf_cast_str2 = """
# import numpy as np
# import tensorflow as tf
# arr_64 = np.random.rand(3, 2)
# test = tf.convert_to_tensor(arr_64.astype(np.float32))
# """
# tf_cast_time = timeit.timeit(tf_cast_str, number=1000)
# tf_cast_time2 = timeit.timeit(tf_cast_str2, number=1000)

# # Time torch casting
# torch_cast_str = """
# import numpy as np
# import torch
# arr_64 = np.random.rand(3, 2)
# test = torch.from_numpy(arr_64).float()
# """
# torch_cast_str_2 = """
# import numpy as np
# import torch
# arr_64 = np.random.rand(3, 2)
# torch.from_numpy(arr_64.astype(np.float32))
# """
# torch_cast_time = timeit.timeit(torch_cast_str, number=1000)
# torch_cast_time2 = timeit.timeit(torch_cast_str, number=1000)

# print(f"np time: {np_cast_time}")
# print(f"tensorflow time: {tf_cast_time}")
# print(f"tensorflow time2: {tf_cast_time2}")
# print(f"torch time: {torch_cast_time}")
# print(f"torch time 2: {torch_cast_time2}")
# print("end")

# == Using spyder profiler

# Create dummy array
arr_64 = np.random.rand(3, 2)

# for i in range(0, 10000000):
for i in range(0, 1000000):
    # - Numpy
    # np_casted = arr_64.astype(np.float32)  # NOTE: Fastest
    
    # # - Tensorflow
    tf_casted = tf.convert_to_tensor(arr_64)  # NOTE: Slower
    tf_casted = tf.cast(tf_casted, dtype=tf.float32)
    # tf_casted = tf.convert_to_tensor(arr_64, dtype=tf.float32)  # NOTE: Slower
    # tf_casted2 = tf.convert_to_tensor(arr_64.astype(np.float32))  # NOTE: Faster
    
    # # - Torch
    # torch_casted = torch.from_numpy(arr_64).float()  # NOTE: Fast
    # torch_casted2 = torch.from_numpy(arr_64.astype(np.float32))  # NOTE: Equally fast
