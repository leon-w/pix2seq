from einops import rearrange

import torch
import torch.nn as nn

# def get_angles_tf(pos, i, dim):
#   angle_rates = 1 / tf.pow(10000., tf.cast(2 * (i//2), tf.float32) / dim)
#   return tf.cast(pos, tf.float32) * tf.cast(angle_rates, tf.float32)


# def positional_encoding_tf(coords, dim):
#   """coords in (bsz, size), return (bsz, size, dim)."""
#   angle_rads = get_angles_tf(tf.expand_dims(coords, -1),
#                           tf.range(dim)[tf.newaxis, tf.newaxis, :],
#                           dim)

#   # apply sin to even indices in the array; 2i
#   angle_rads1 = tf.sin(angle_rads[:, :, 0::2])

#   # apply cos to odd indices in the array; 2i+1
#   angle_rads2 = tf.cos(angle_rads[:, :, 1::2])

#   pos_encoding = tf.concat([angle_rads1, angle_rads2], -1)

#   return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, dim):
    angle_rates = 1 / torch.pow(10000.0, 2 * (i // 2) / dim)
    return pos.float() * angle_rates.float()


def positional_encoding(coords, dim):
    angle_rads = get_angles(
        rearrange(coords, "b s -> b s 1"),
        rearrange(torch.arange(dim), "d -> 1 1 d"),
        dim,
    )

    # apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[..., 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[..., 1::2])

    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

    return pos_encoding.float()


# class ScalarEmbedding_tf(tf.keras.layers.Layer):
#   """Scalar embedding layers.

#   Assume the first input dim to be time, and rest are optional features.
#   """

#   def __init__(self, dim, scaling, expansion=4, **kwargs):
#     super().__init__(**kwargs)
#     self.scalar_encoding = lambda x: positional_encoding_tf(x*scaling, dim)
#     self.dense_0 = tf.keras.layers.Dense(
#         dim * expansion,
#         kernel_initializer=get_variable_initializer(1.),
#         name='dense0')
#     self.dense_1 = tf.keras.layers.Dense(
#         dim * expansion,
#         kernel_initializer=get_variable_initializer(1.),
#         name='dense1')

#   def call(self, x, last_swish=True, normalize=False):
#     y = None
#     if x.shape.rank > 1:
#       assert x.shape.rank == 2
#       x, y = x[..., 0], x[..., 1:]
#     x = self.scalar_encoding(x)[0]
#     if normalize:
#       x_mean = tf.reduce_mean(x, -1, keepdims=True)
#       x_std = tf.math.reduce_std(x, -1, keepdims=True)
#       x = (x - x_mean) / x_std
#     x = tf.nn.silu(self.dense_0(x))
#     x = x if y is None else tf.concat([x, y], -1)
#     x = self.dense_1(x)
#     return tf.nn.silu(x) if last_swish else x


class ScalarEmbedding(nn.Module):
    """Scalar embedding layers.

    Assume the first input dim to be time, and rest are optional features.
    """

    def __init__(self, dim, scaling, expansion=4):
        super().__init__()
        self.scalar_encoding = lambda x: positional_encoding(x * scaling, dim)
        self.dense_0 = nn.Linear(dim, dim * expansion)
        self.dense_1 = nn.Linear(dim * expansion, dim * expansion)

    def forward(self, x, last_swish=True, normalize=False):
        y = None
        if x.dim() > 1:
            assert x.dim() == 2
            x, y = x[..., 0], x[..., 1:]
        x = self.scalar_encoding(x)[0]
        if normalize:
            x_mean = torch.mean(x, -1, keepdim=True)
            x_std = torch.std(x, -1, keepdim=True)
            x = (x - x_mean) / x_std
        x = nn.functional.silu(self.dense_0(x))
        x = x if y is None else torch.cat([x, y], -1)
        x = self.dense_1(x)
        return nn.functional.silu(x) if last_swish else x
