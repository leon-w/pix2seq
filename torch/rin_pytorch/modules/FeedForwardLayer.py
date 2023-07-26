import torch
import torch.nn as nn

from ..utils.initializer import initialize_linear


class FeedForwardLayer(nn.Module):
    # done
    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        drop_units=0.1,
        use_ln=False,
        ln_scale_shift=False,
    ):
        super().__init__()
        self.dense1 = nn.Linear(dim, dim_hidden)
        initialize_linear(self.dense1)

        self.dropout = nn.Dropout(drop_units)

        self.dense2 = nn.Linear(dim_hidden, dim)
        initialize_linear(self.dense2)

        if use_ln:
            self.ln = nn.LayerNorm(dim_hidden, eps=1e-6, elementwise_affine=ln_scale_shift)
        else:
            self.ln = nn.Identity()

    # done
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = nn.functional.gelu(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
