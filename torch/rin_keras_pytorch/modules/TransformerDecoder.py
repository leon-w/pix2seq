import keras_core as keras

import torch

from .TransformerDecoderLayer import TransformerDecoderLayer


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
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
        self.num_layers = num_layers
        self.dec_layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim,
                    mlp_ratio,
                    num_heads,
                    drop_path,
                    drop_units,
                    drop_att,
                    dim_x_att=dim_x_att,
                    self_attention=self_attention,
                    cross_attention=cross_attention,
                    use_mlp=use_mlp,
                    use_enc_ln=use_enc_ln,
                    use_ffn_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc):
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc)

        return x
