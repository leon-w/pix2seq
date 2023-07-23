from DropPath import DropPath
from MLP import MLP

import torch.nn as nn

# # TENSORFLOW IMPLEMENTATION
# class TransformerEncoderLayer_tf(tf.keras.layers.Layer):
#   def __init__(self,
#                dim,
#                mlp_ratio,
#                num_heads,
#                drop_path=0.1,
#                drop_units=0.1,
#                drop_att=0.,
#                self_attention=True,
#                use_ffn_ln=False,
#                ln_scale_shift=True,
#                **kwargs):
#     super(TransformerEncoderLayer_tf, self).__init__(**kwargs)
#     self.self_attention = self_attention
#     if self_attention:
#       self.mha_ln = tf.keras.layers.LayerNormalization(
#           epsilon=1e-6,
#           center=ln_scale_shift,
#           scale=ln_scale_shift,
#           name='mha/ln')
#       self.mha = tf.keras.layers.MultiHeadAttention(
#           num_heads, dim // num_heads, dropout=drop_att, name='mha')
#     self.mlp = MLP_tf(1, dim, mlp_ratio, drop_path, drop_units,
#                    use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
#                    name='mlp')
#     self.dropp = DropPath_tf(drop_path)

#   def call(self, x, mask, training):
#     # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
#     if self.self_attention:
#       x_ln = self.mha_ln(x)
#       x_residual = self.mha(x_ln, x_ln, x_ln, mask, training=training)
#       x = x + self.dropp(x_residual, training)
#     x = self.mlp(x, training)
#     return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        self_attention=True,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.self_attention = self_attention
        if self_attention:
            self.mha_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.mha = nn.MultiheadAttention(dim, num_heads, dropout=drop_att)
        self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units, use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift)
        self.dropp = DropPath(drop_path)

    def forward(self, x, mask):
        # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
        if self.self_attention:
            x_ln = self.mha_ln(x)
            x_residual = self.mha(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False)[0]
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return x
