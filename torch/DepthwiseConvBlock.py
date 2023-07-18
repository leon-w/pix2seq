import math

from einops import rearrange

import torch
import torch.nn as nn

# class DepthwiseConvBlock_tf(tf.keras.layers.Layer):
#   """Depthwise conv followed by pointwise/1x1 conv."""
#   def __init__(self,
#                out_dim,
#                kernel_size,
#                dropout_rate=0.,
#                **kwargs):
#     super().__init__(**kwargs)
#     self._out_dim = out_dim
#     self._kernel_size = kernel_size
#     self._dropout_rate = dropout_rate

#   def build(self, input_shapes):
#     input_dim = self._out_dim if input_shapes is None else input_shapes[-1]
#     self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
#         kernel_size=self._kernel_size,
#         padding='SAME',
#         use_bias=True,
#         kernel_initializer=get_variable_initializer(1.0),
#         name='depthwise_conv')
#     self.gn = get_norm(
#         'group_norm',
#         num_groups=min(input_dim // 4, 32),
#         name='gn')
#     self.dropout = tf.keras.layers.Dropout(self._dropout_rate)
#     self.pointwise_conv = tf.keras.layers.Conv2D(
#         filters=self._out_dim,
#         kernel_size=[1, 1],
#         strides=[1, 1],
#         padding='SAME',
#         use_bias=True,
#         kernel_initializer=get_variable_initializer(1e-10),
#         name='pointwise_conv')

#   def call(self, x, training, size=None):
#     """call function.

#     Args:
#       x: `Tensor` of (bsz, h, w, c) or (bsz, seqlen, c).
#       training: `Boolean` indicator.
#       size: set to None if x is an 4d image tensor, otherwise set size=h*w,
#         where seqlen=size+extra, and conv is performed only on the first part.

#     Returns:
#       `Tensor` of the same shape as input x.
#     """
#     x_skip = x
#     if size is not None:  # Resize sequence into an image for 2d conv.
#       x_skip = x[:, :size]
#       x_remain = x[:, size:]
#       height = width = int(math.sqrt(size))
#       x = tf.reshape(x_skip, [tf.shape(x)[0], height, width, tf.shape(x)[-1]])
#     x = tf.nn.silu(self.gn(self.depthwise_conv(x)))
#     x = self.dropout(x, training=training)
#     x = self.pointwise_conv(x)
#     # TODO(iamtingchen): consider unet style ordering of gn&conv.
#     # x = tf.nn.silu(self.gn1(x))
#     # x = self.depthwise_conv(x)
#     # x = tf.nn.silu(self.gn2(x))
#     # x = self.dropout(x, training=training)
#     # x = self.pointwise_conv(x)
#     # TODO(iamtingchen): consider transformer/normer style ordering of gn&conv.
#     # x = tf.nn.silu(self.depthwise_conv(self.gn1(x)))
#     # x = self.dropout(x, training=training)
#     # x = self.pointwise_conv(self.gn2(x))
#     if size is not None:
#       x = x_skip + tf.reshape(x, [tf.shape(x)[0], size, tf.shape(x)[-1]])
#       x = tf.concat([x, x_remain], 1)
#     else:
#       x = x_skip + x
#     return x


class DepthwiseConvBlock(nn.Module):
    def __init__(self, out_dim, kernel_size, dropout_rate=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.depthwise_conv = nn.Conv2d(
            in_channels=self.out_dim,
            out_channels=self.out_dim,
            kernel_size=self.kernel_size,
            padding="SAME",
            groups=self.out_dim,
            bias=True,
        )
        self.gn = nn.GroupNorm(num_groups=min(self.out_dim // 4, 32), num_channels=self.out_dim)
        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=1, stride=1, padding="SAME", bias=True
        )

    def forward(self, x, size=None):
        x_skip = x
        if size is not None:
            x_skip = x[:, :size]
            x_remain = x[:, size:]
            height = width = int(math.sqrt(size))
            x = x_skip.view(-1, self.out_dim, height, width)
        x = self.gn(self.depthwise_conv(x))
        x = nn.functional.silu(x)
        x = self.dropout(x)
        x = self.pointwise_conv(x)
        if size is not None:
            x = x_skip + x.view(-1, size, self.out_dim)
            x = torch.cat([x, x_remain], 1)
        else:
            x = x_skip + x
        return x
