import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class SquashBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name
        )

    def _forward(self, x):
        return tf.nn.tanh(x)
        # return x

    def _inverse(self, y):
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2.0 * (np.log(2.0) - x - tf.nn.softplus(-2.0 * x))
