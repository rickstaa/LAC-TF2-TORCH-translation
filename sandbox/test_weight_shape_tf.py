import tensorflow as tf
import torch

tf.executing_eagerly()

x = [[2.0]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

# Create simple net
torch.manual_seed(0)
w_init_net_0 = tf.constant_initializer(torch.randn((2, 64)).numpy())
b_init_net_0 = tf.constant_initializer(torch.randn((64)).numpy())
net_0 = tf.compat.v1.layers.dense(
    2,
    64,
    activation=tf.nn.relu,
    name="l1",
    bias_initializer=b_init_net_0,
    kernel_initializer=w_init_net_0,
    trainable=True,
)  # 原始是30
print("ets")
