"""Script used to compare the execution time difference when running similar code in
tensorflow 1.15 and tf2.3.
"""

import time
import os
from packaging import version

import tensorflow as tf

# Script settings
USE_GPU = False
N_STEPS = 1e5

# Disable GPU if requested
if not USE_GPU:  # NOTE: This works in both TF115 and tf2
    # tf.config.set_visible_devices([], "GPU")
    if version.parse(tf.__version__) > version.parse("1.15.4"):
        tf.config.set_visible_devices([], "GPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Tensorflow is using CPU")
else:
    print("Tensorflow is using GPU")

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Create session
sess = tf.compat.v1.Session()

# Setup placeholders
t = 0  # Time
x = tf.compat.v1.placeholder(tf.float32, [None, 1], "x")
y = tf.math.sin(x, name=None)

# Run session in loop
print("===TF115 speed test script===")
print(f"Running a tf session {int(N_STEPS)} times to test the execution speed.")
print("Starting sess run loop.")
t1 = time.time()
for i in range(0, int(N_STEPS)):

    # Run result in session
    y_val = sess.run(y, feed_dict={x: [[t]]})

    # Increment time
    t += 1

    # print current step
    if i % 10000 == 0:
        print(f"Performed {i} steps.")

# Print end result
print("sess run loop stopped.")
print("Running time: ", time.time() - t1)
