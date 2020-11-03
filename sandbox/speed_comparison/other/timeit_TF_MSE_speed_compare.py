"""Small script to see which (Tensorflow) method of calculating the Mean Squared Root error is
faster.

Conclusion:
    There is no difference betwen the two methods.
"""

import timeit

# Script settings
N_SAMPLE = int(1e6)

# Test functional MSE speed
tf_setup_code = """
import tensorflow as tf
batch_size=256
l_backup = tf.random.uniform((batch_size, 1), dtype=tf.float32)
l1 = tf.random.uniform((batch_size, 1), dtype=tf.float32)
@tf.function
def sample_function():
    l_error = 0.5 * tf.compat.v1.losses.mean_squared_error(
        labels=l_backup, predictions=l1
    )
"""
tf_sample_code = """
sample_function()
"""
method_1_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

# Test tf2 functional MSE speed
tf_setup_code = """
import tensorflow as tf
batch_size=256
l_backup = tf.random.uniform((batch_size, 1), dtype=tf.float32)
l1 = tf.random.uniform((batch_size, 1), dtype=tf.float32)
mse=tf.keras.losses.MeanSquaredError()
@tf.function
def sample_function():
    l_error = 0.5 * mse(
        l_backup, l1
    )
"""
tf_sample_code = """
sample_function()
"""
method_2_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)
# Test manual MSE speed
tf_setup_code = """
import tensorflow as tf
batch_size=256
l_backup = tf.random.uniform((batch_size, 1), dtype=tf.float32)
l1 = tf.random.uniform((batch_size, 1), dtype=tf.float32)
@tf.function
def sample_function():
    l_error = 0.5 * tf.reduce_mean(
        (l1 - l_backup) ** 2
    )
    """
tf_sample_code = """
sample_function()
"""
method_3_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)


# Print results
print("\nTest MSE methods:")
print(f"- Functional tf1.5 MSE time: {method_1_time} s")
print(f"- functional tf 2.0 MSE time: {method_2_time} s")
print(f"- Manual MSE: {method_3_time} s")
