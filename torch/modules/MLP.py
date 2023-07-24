from DropPath import DropPath
from FeedForwardLayer import FeedForwardLayer

import torch
import torch.nn as nn


class MLP(nn.Module):
    # done
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

        self.mlp_layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()

        for _ in range(num_layers):
            self.mlp_layers.append(
                FeedForwardLayer(
                    dim=dim,
                    dim_hidden=dim * mlp_ratio,
                    drop_units=drop_units,
                    use_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
            )
            self.layernorms.append(
                nn.LayerNorm(
                    dim,
                    eps=1e-6,
                    elementwise_affine=ln_scale_shift,
                )
            )

        self.dropp = DropPath(drop_path)

    # done
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mlp, ln in zip(self.mlp_layers, self.layernorms):
            x_residual = self.dropp(mlp(ln(x)))
            x = x + x_residual
        return x
