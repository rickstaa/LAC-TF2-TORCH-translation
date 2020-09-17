"""Small script to find out how the seed behavior differs in eager and graph mode.
https://www.tensorflow.org/api_docs/python/tf/compat/v1/set_random_seed
https://www.tensorflow.org/guide/random_numbers
https://github.com/tensorflow/tensorflow/issues/35739
https://github.com/OlafenwaMoses/ImageAI/issues/400
"""

import random
import os
import numpy as np
import tensorflow as tf


##############################################
# Disable eager execution ####################
##############################################

# # Set seeds
tf.compat.v1.disable_eager_execution()
os.environ["PYTHONHASHSEED"] = str(0)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
tf.random.set_seed(0)
tf.compat.v1.random.set_random_seed(0)
np.random.seed(0)
random.seed(0)

# # # First with individual components
# # # a = tf.random.uniform([1], seed=0)
# # # # tf.random.set_seed(0)
# # # a2 = tf.random.uniform([1], seed=0)
# # # with tf.compat.v1.Session() as sess1:
# # #     print(sess1.run(a))
# # #     print(sess1.run(a2))

# # # # Now with a batch
# # # # batch = {
# # # #     "s": tf.random.uniform((2, 2), seed=0),
# # # #     "a": tf.random.uniform((2, 2), seed=0),
# # # #     "r": tf.random.uniform((2, 1), seed=0),
# # # #     "terminal": tf.zeros((2, 1)),
# # # #     "s_": tf.random.uniform((2, 2), seed=0),
# # # # }
# # # tf.random.set_seed(0)
# # # batch = {
# # #     "s": tf.random.uniform((2, 2)),
# # #     "a": tf.random.uniform((2, 2)),
# # #     "r": tf.random.uniform((2, 1)),
# # #     "terminal": tf.zeros((2, 1)),
# # #     "s_": tf.random.uniform((2, 2)),
# # # }
# # # with tf.compat.v1.Session() as sess1:
# # #     batch_res = sess1.run(batch)
# # # print("jan")

# # Implicit seeding
# # NOTE: Doesn't work (Other result than eager)
# sess = tf.compat.v1.Session()
# tf.random.set_seed(0)
# s_tmp = tf.random.uniform((2, 2))
# a_tmp = tf.random.uniform((2, 2))
# r_tmp = tf.random.uniform((2, 1))
# terminal_tmp = tf.zeros((2, 1))
# s_target_tmp = tf.random.uniform((2, 2))
# tf.random.set_seed(0)
# print(sess.run(s_tmp))
# print(sess.run(a_tmp))
# print(sess.run(r_tmp))
# print(sess.run(terminal_tmp))
# print(sess.run(s_target_tmp))
# # print("jan")

# ## Explicit seeding
# # tf.random.set_seed(0)
# # s_tmp = tf.random.uniform((2, 2), seed=0)
# # a_tmp = tf.random.uniform((2, 2), seed=1)
# # r_tmp = tf.random.uniform((2, 1), seed=2)
# # terminal_tmp = tf.zeros((2, 1))
# # s_target_tmp = tf.random.uniform((2, 2), seed=3)
# # with tf.compat.v1.Session() as sess1:
# #     print(sess1.run(s_tmp))
# #     print(sess1.run(a_tmp))
# #     print(sess1.run(r_tmp))
# #     print(sess1.run(terminal_tmp))
# #     print(sess1.run(s_target_tmp))
# # # print("jan")

#############################################
# Enable eager execution ####################
#############################################

# # Set seeds
# os.environ["PYTHONHASHSEED"] = str(0)
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
# tf.random.set_seed(0)
# tf.compat.v1.random.set_random_seed(0)
# np.random.seed(0)
# random.seed(0)

# # First with individual components
# # a = tf.random.uniform([1], seed=0)
# # # a = tf.random.uniform([1], seed=0)
# # # tf.compat.v1.random.set_random_seed(1234)5
# # tf.random.set_seed(0)
# # a2 = tf.random.uniform([1], seed=0)
# # print(a)
# # print(a2)

# # # Now with a batch
# # tf.random.set_seed(0)
# # # batch = {
# # #     "s": tf.random.uniform((10, 2), seed=0),
# # #     "a": tf.random.uniform((10, 2), seed=0),
# # #     "r": tf.random.uniform((10, 1), seed=0),
# # #     "terminal": tf.zeros((10, 1)),
# # #     "s_": tf.random.uniform((10, 2), seed=0),
# # # }
# # batch = {
# #     "s": tf.random.uniform((2, 2)),
# #     "a": tf.random.uniform((2, 2)),
# #     "r": tf.random.uniform((2, 1)),
# #     "terminal": tf.zeros((2, 1)),
# #     "s_": tf.random.uniform((2, 2)),
# # }

# # Implicit seeding
# tf.random.set_seed(0)
# s_tmp = tf.random.uniform((2, 2))
# a_tmp = tf.random.uniform((2, 2))
# r_tmp = tf.random.uniform((2, 1))
# terminal_tmp = tf.zeros((2, 1))
# s_target_tmp = tf.random.uniform((2, 2))
# print(s_tmp)
# print(a_tmp)
# print(r_tmp)
# print(terminal_tmp)
# print(s_target_tmp)
# print("jan")

# # # Explicit seeding
# # tf.random.set_seed(0)
# # s_tmp = tf.random.uniform((2, 2), seed=0)
# # a_tmp = tf.random.uniform((2, 2), seed=1)
# # r_tmp = tf.random.uniform((2, 1), seed=2)
# # terminal_tmp = tf.zeros((2, 1))
# # s_target_tmp = tf.random.uniform((2, 2), seed=3)
# # print(s_tmp)
# # print(a_tmp)
# # print(r_tmp)
# # print(terminal_tmp)
# # print(s_target_tmp)
# # print("jan")
