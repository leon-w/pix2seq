import torch
import torch.nn as nn

# # TENSORFLOW IMPLEMENTATION
# class DropPath_tf(tf.keras.layers.Layer):
#     """For stochastic depth."""

#     def __init__(self, drop_rate=0.0, **kwargs):
#         """Initializes a drop path layer."""
#         super(DropPath_tf, self).__init__(**kwargs)
#         self._drop_rate = drop_rate
#         if self._drop_rate < 0 or self._drop_rate >= 1.0:
#             raise ValueError("drop_rate {} is outside [0, 1)".format(self._drop_rate))

#     def call(self, x, training=False):
#         """Performs a forward pass.

#         Args:
#         x: An input tensor of type tf.Tensor with shape [batch, height,
#             width, channels].
#         training: A boolean flag indicating whether training behavior should be
#             used (default: False).

#         Returns:
#         The output tensor.
#         """
#         if self._drop_rate == 0.0 or not training:
#             return x

#         keep_rate = 1.0 - self._drop_rate
#         xshape = tf.shape(x)
#         drop_mask_shape = [xshape[0]] + [1] * (len(xshape) - 1)
#         drop_mask = keep_rate + tf.random.uniform(drop_mask_shape, dtype=x.dtype)
#         drop_mask = tf.math.divide(tf.floor(drop_mask), keep_rate)

#         return x * drop_mask


# PYTORCH IMPLEMENTATION
class DropPath(nn.Module):
    def __init__(self, drop_rate=0.0):
        super().__init__()

        assert 0.0 <= drop_rate <= 1.0
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.drop_rate == 0.0 or not self.training:
            return x

        keep_rate = 1.0 - self.drop_rate
        drop_mask_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
        drop_mask = keep_rate + torch.rand(drop_mask_shape, dtype=x.dtype, device=x.device)
        drop_mask = torch.floor(drop_mask) / keep_rate

        return x * drop_mask
