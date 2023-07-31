import keras_core as keras

import torch

from .TransformerEncoderLayer import TransformerEncoderLayer


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        self_attention=True,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.enc_layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    mlp_ratio,
                    num_heads,
                    drop_path,
                    drop_units,
                    drop_att,
                    self_attention=self_attention,
                    use_ffn_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x
