import torch
import torch.nn as nn

from .utils.initializer import initialize_variable_variance_scaling
from .utils.pos_embedding import positional_encoding


class ScalarEmbedding(nn.Module):
    # done
    def __init__(
        self,
        dim: int,
        scaling: float | torch.Tensor,
        expansion=4,
    ):
        super().__init__()
        self.scalar_encoding = lambda x: positional_encoding(x * scaling, dim)
        self.dense_0 = nn.Linear(dim, dim * expansion)
        initialize_variable_variance_scaling(self.dense_0.weight, scale=1.0)
        self.dense_1 = nn.Linear(dim * expansion, dim * expansion)
        initialize_variable_variance_scaling(self.dense_1.weight, scale=1.0)

    # done
    def forward(
        self,
        x: torch.Tensor,
        last_swish=True,
        normalize=False,
    ) -> torch.Tensor:
        assert x.ndim == 1
        x = self.scalar_encoding(x)[0]
        if normalize:
            x_mean = torch.mean(x, -1, keepdim=True)
            x_std = torch.std(x, -1, keepdim=True)
            x = (x - x_mean) / x_std
        x = nn.functional.silu(self.dense_0(x))
        x = self.dense_1(x)
        return nn.functional.silu(x) if last_swish else x
