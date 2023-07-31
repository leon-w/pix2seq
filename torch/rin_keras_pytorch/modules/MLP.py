import keras_core as keras

import torch

from .DropPath import DropPath
from .FeedForwardLayer import FeedForwardLayer


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        mlp_ratio: int,
        drop_path=0.1,
        drop_units=0.0,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_layers = torch.nn.ModuleList(
            [
                FeedForwardLayer(
                    dim,
                    dim * mlp_ratio,
                    drop_units,
                    use_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorms = torch.nn.ModuleList(
            [
                keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    center=ln_scale_shift,
                    scale=ln_scale_shift,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropp = DropPath(drop_path)

    def forward(self, x):
        for i in range(self.num_layers):
            x_residual = self.mlp_layers[i](self.layernorms[i](x))
            x = x + self.dropp(x_residual)
        return x
