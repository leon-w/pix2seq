import keras_core as keras

import torch

from .DropPath import DropPath
from .MLP import MLP


class TransformerEncoderLayer(torch.nn.Module):  # pylint: disable=missing-docstring
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
        **kwargs
    ):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.self_attention = self_attention
        if self_attention:
            self.mha_ln = keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=ln_scale_shift,
                scale=ln_scale_shift,
            )
            self.mha = keras.layers.MultiHeadAttention(
                num_heads,
                dim // num_heads,
                dropout=drop_att,
            )
        self.mlp = MLP(
            1,
            dim,
            mlp_ratio,
            drop_path,
            drop_units,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
        )
        self.dropp = DropPath(drop_path)

    def forward(self, x):
        if self.self_attention:
            x_ln = self.mha_ln(x)
            x_residual = self.mha(x_ln, x_ln, x_ln, training=self.training)
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return x
