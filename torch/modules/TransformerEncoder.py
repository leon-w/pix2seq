from TransformerEncoderLayer import TransformerEncoderLayer

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    # done
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
        self.enc_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    drop_path=drop_path,
                    drop_units=drop_units,
                    drop_att=drop_att,
                    self_attention=self_attention,
                    use_ffn_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
                for _ in range(num_layers)
            ]
        )

    # done
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.enc_layers:
            x = layer(x)
        return x
