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


# def get_1d_position_codes_tf(seqlen, out_dim, normalization_max=6.2831852):
#   """Get 2d positional embedding with sin/cos codes.

#   Args:
#     seqlen: a `int` specifying the length of the sequence.
#     out_dim: a `int` specifying the output dimension of the encoding.
#     normalization_max: normalize coordinates between [0, normalization_max].
#       If None, raw coordinates from 0 to seqlen will be used.

#   Returns:
#     positional code of shape (1, seqlen, out_dim)
#   """
#   coords = tf.cast(tf.range(seqlen), tf.float32)
#   if normalization_max is not None:
#     coords = coords / (seqlen - 1) * normalization_max
#   coords = positional_encoding_tf(coords, out_dim)
#   return coords


def get_1d_position_codes(seqlen, out_dim, normalization_max=6.2831852):
    coords = torch.arange(seqlen, dtype=torch.float32)
    if normalization_max is not None:
        coords = coords / (seqlen - 1) * normalization_max
    coords = positional_encoding(coords, out_dim)
    return coords


# def get_2d_position_codes_tf(height, width, out_dim, normalization_max=6.2831852):
#   """Get 2d positional embedding with sin/cos codes.

#   Args:
#     height: a `int` specifying the height of the 2d image / feature map.
#     width: a `int` specifying the width of the 2d image / feature map.
#     out_dim: a `int` specifying the output dimension of the encoding.
#       Must be divisible by 2.
#     normalization_max: normalize coordinates between [0, normalization_max].
#       If None, raw coordinates from 0 to height/width will be used.

#   Returns:
#     positional code of shape (1, height, width, out_dim)
#   """
#   y_coords = tf.cast(tf.range(height), tf.float32)
#   if normalization_max is not None:
#     y_coords = (
#         y_coords / tf.cast(height - 1, dtype=tf.float32) * normalization_max)
#   y_coords = positional_encoding_tf(y_coords, out_dim//2)
#   y_coords = tf.expand_dims(y_coords, 2)
#   y_coords = tf.concat([y_coords, tf.zeros_like(y_coords)], -1)

#   x_coords = tf.cast(tf.range(width), tf.float32)
#   if normalization_max is not None:
#     x_coords = (
#         x_coords / tf.cast(width - 1, dtype=tf.float32) * normalization_max)
#   x_coords = positional_encoding_tf(x_coords, out_dim//2)
#   x_coords = tf.expand_dims(x_coords, 1)
#   x_coords = tf.concat([tf.zeros_like(x_coords), x_coords], -1)

#   return y_coords + x_coords


def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
    y_coords = torch.arange(height, dtype=torch.float32)
    if normalization_max is not None:
        y_coords = y_coords / torch.tensor(height - 1, dtype=torch.float32) * normalization_max
    y_coords = positional_encoding(y_coords, out_dim // 2)
    y_coords = y_coords.unsqueeze(2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = torch.arange(width, dtype=torch.float32)
    if normalization_max is not None:
        x_coords = x_coords / torch.tensor(width - 1, dtype=torch.float32) * normalization_max
    x_coords = positional_encoding(x_coords, out_dim // 2)
    x_coords = x_coords.unsqueeze(1)
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords


# def add_seq_pos_emb_tf(self, pos_encoding, max_seq_len, dim,
#                     name_prefix=None, initializer=None):
#   """Add seq_pos_emb variable/tensor to model instance referenced by `self`."""
#   if name_prefix is None:
#     name_prefix = self.name
#   if initializer is None:
#     initializer = get_variable_initializer()
#   if pos_encoding == 'learned':
#     self.seq_pos_emb = self.add_weight(
#         shape=(max_seq_len, dim), initializer=initializer,
#         name='%s/seq_pos_embedding' % name_prefix)
#   elif pos_encoding == 'sin_cos':
#     sin_cos = get_1d_position_codes_tf(
#         max_seq_len, dim, normalization_max=6.2831852)
#     self.seq_pos_emb = tf.reshape(sin_cos, [max_seq_len, dim])
#   else:
#     raise ValueError('Unknown pos encoding %s' % pos_encoding)


def add_seq_pos_emb(self: nn.Module, pos_encoding, max_seq_len, dim, initializer=None):
    """Add seq_pos_emb variable/tensor to model instance referenced by `self`."""
    if initializer is None:
        initializer = nn.init.xavier_uniform_
    if pos_encoding == "learned":
        seq_pos_emb = nn.Parameter(initializer(torch.empty(max_seq_len, dim)), requires_grad=True)
        self.register_parameter("seq_pos_emb", seq_pos_emb)
    elif pos_encoding == "sin_cos":
        sin_cos = get_1d_position_codes(max_seq_len, dim, normalization_max=6.2831852)
        self.seq_pos_emb = torch.reshape(sin_cos, [max_seq_len, dim])
    else:
        raise ValueError(f"Unknown pos encoding pos_encoding")


def add_vis_pos_emb(
    self,
    pos_encoding,
    n_rows,
    n_cols,
    dim,
    initializer=None,
    return_only=False,
    normalization_max=6.2831852,
):
    """Add vis_pos_emb variable/tensor to model instance referenced by `self`."""
    if initializer is None:
        initializer = nn.init.xavier_uniform_
    if pos_encoding == "learned":
        vis_pos_emb = nn.Parameter(initializer(torch.empty(n_rows * n_cols, dim)), requires_grad=True)
        self.register_parameter("vis_pos_emb", vis_pos_emb)
    elif pos_encoding == "sin_cos":
        if n_rows == 1 or n_cols == 1:
            sin_cos = get_1d_position_codes(n_rows * n_cols, dim, normalization_max=normalization_max)
        else:
            sin_cos = get_2d_position_codes(n_rows, n_cols, dim, normalization_max=normalization_max)
        vis_pos_emb = sin_cos.view(n_rows * n_cols, dim)

        if not return_only:
            self.vis_pos_emb = vis_pos_emb
    else:
        raise ValueError(f"Unknown pos encoding pos_encoding")

    return vis_pos_emb
