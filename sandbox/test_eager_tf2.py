import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Disable eager
tf.compat.v1.disable_eager_execution()

# Set random seed
os.environ["PYTHONHASHSEED"] = str(0)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
np.random.seed(0)
tf.compat.v1.reset_default_graph()
random.seed(0)
tf.random.set_seed(0)

# Create session
sess = tf.compat.v1.Session()

# Sample from distribution
seed = tfp.util.SeedStream(0, salt="random_beta")
base_distribution = tfp.distributions.MultivariateNormalDiag(
    loc=tf.zeros(2), scale_diag=tf.ones(2)
)
epsilon = base_distribution.sample(2, seed=seed())
epsilon = sess.run(epsilon)
print(epsilon)
