from DropPath import DropPath
from MLP import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F

# # TENSORFLOW IMPLEMENTATION
# class TransformerDecoderLayer_tf(tf.keras.layers.Layer):
#   def __init__(self,
#                dim,
#                mlp_ratio,
#                num_heads,
#                drop_path=0.1,
#                drop_units=0.1,
#                drop_att=0.,
#                dim_x_att=None,
#                self_attention=True,
#                cross_attention=True,
#                use_mlp=True,
#                use_enc_ln=False,
#                use_ffn_ln=False,
#                ln_scale_shift=True,
#                **kwargs):
#     super(TransformerDecoderLayer_tf, self).__init__(**kwargs)
#     self.self_attention = self_attention
#     self.cross_attention = cross_attention
#     self.use_mlp = use_mlp
#     if self_attention:
#       self.self_ln = tf.keras.layers.LayerNormalization(
#           epsilon=1e-6,
#           center=ln_scale_shift,
#           scale=ln_scale_shift,
#           name='self_mha/ln')
#       self.self_mha = tf.keras.layers.MultiHeadAttention(
#           num_heads, dim // num_heads, dropout=drop_att, name='self_mha')
#     if cross_attention:
#       self.cross_ln = tf.keras.layers.LayerNormalization(
#           epsilon=1e-6,
#           center=ln_scale_shift,
#           scale=ln_scale_shift,
#           name='cross_mha/ln')
#       if use_enc_ln:
#         self.enc_ln = tf.keras.layers.LayerNormalization(
#             epsilon=1e-6,
#             center=ln_scale_shift,
#             scale=ln_scale_shift,
#             name='cross_mha/enc_ln')
#       else:
#         self.enc_ln = lambda x: x
#       dim_x_att = dim if dim_x_att is None else dim_x_att
#       self.cross_mha = tf.keras.layers.MultiHeadAttention(
#           num_heads, dim_x_att // num_heads, dropout=drop_att, name='cross_mha')
#     if use_mlp:
#       self.mlp = MLP_tf(1, dim, mlp_ratio, drop_path, drop_units,
#                      use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
#                      name='mlp')
#     self.dropp = DropPath_tf(drop_path)

#   def call(self, x, enc, cache, mask_self, mask_cross, training):
#     """x in (bsz, seq, d), enc in (bsz, seq', d)."""
#     x_for_cache = []
#     if self.self_attention:
#       x_for_cache = x_ln = kv_ln = self.self_ln(x)
#       if cache is not None:  # Augment kv_ln with cache in (bsz, c_size, d).
#         q_size, k_size = tf.shape(x)[1], tf.shape(cache)[1]
#         mask_self = tf.concat([tf.ones([1, 1, q_size, k_size]), mask_self], -1)
#         kv_ln = tf.concat([cache, x_ln], axis=1)
#       x_res = self.self_mha(x_ln, kv_ln, kv_ln, mask_self, training=training)
#       x = x + self.dropp(x_res, training)
#     if self.cross_attention:
#       x_ln = self.cross_ln(x)
#       enc = self.enc_ln(enc)
#       x_res = self.cross_mha(x_ln, enc, enc, mask_cross, training=training)
#       x = x + self.dropp(x_res, training)
#     if self.use_mlp:
#       x = self.mlp(x, training)
#     return x, x_for_cache


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        dim_x_att=None,
        self_attention=True,
        cross_attention=True,
        use_mlp=True,
        use_enc_ln=False,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.use_mlp = use_mlp
        if self_attention:
            self.self_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.self_mha = nn.MultiheadAttention(dim, num_heads, dropout=drop_att)
        if cross_attention:
            self.cross_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.enc_ln = (
                nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift) if use_enc_ln else nn.Identity()
            )
            dim_x_att = dim if dim_x_att is None else dim_x_att
            self.cross_mha = nn.MultiheadAttention(dim_x_att, num_heads, dropout=drop_att)
        if use_mlp:
            self.mlp = MLP(
                1, dim, mlp_ratio, drop_path, drop_units, use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift
            )
        self.dropp = DropPath(drop_path)

    def forward(self, x, enc, cache, mask_self, mask_cross):
        """x in (bsz, seq, d), enc in (bsz, seq', d)."""
        x_for_cache = []
        if self.self_attention:
            x_for_cache = x_ln = kv_ln = self.self_ln(x)
            if cache is not None:  # Augment kv_ln with cache in (bsz, c_size, d).
                q_size, k_size = x.size(1), cache.size(1)
                mask_self = torch.cat([torch.ones([1, 1, q_size, k_size]), mask_self], -1)
                kv_ln = torch.cat([cache, x_ln], dim=1)
            x_res = self.self_mha(x_ln, kv_ln, kv_ln, attn_mask=mask_self, need_weights=False)[0]
            x = x + self.dropp(x_res)
        if self.cross_attention:
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)
            x_res = self.cross_mha(x_ln, enc, enc, attn_mask=mask_cross, need_weights=False)[0]
            x = x + self.dropp(x_res)
        if self.use_mlp:
            x = self.mlp(x)
        return x, x_for_cache
