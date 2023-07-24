import torch
import torch.nn as nn

from .DropPath import DropPath
from .MLP import MLP


class TransformerDecoderLayer(nn.Module):
    # done
    def __init__(
        self,
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
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.use_mlp = use_mlp
        if self_attention:
            self.self_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.self_mha = nn.MultiheadAttention(dim, num_heads, dropout=drop_att)
        if cross_attention:
            self.cross_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            if use_enc_ln:
                self.enc_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            else:
                self.enc_ln = nn.Identity()
            dim_x_att = dim if dim_x_att is None else dim_x_att
            self.cross_mha = nn.MultiheadAttention(dim_x_att, num_heads, dropout=drop_att)
        if use_mlp:
            self.mlp = MLP(
                num_layers=1,
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                drop_units=drop_units,
                use_ffn_ln=use_ffn_ln,
                ln_scale_shift=ln_scale_shift,
            )
        self.dropp = DropPath(drop_path)

    # done
    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
    ) -> torch.Tensor:
        if self.self_attention:
            x_ln = self.self_ln(x)
            x_res, _ = self.self_mha(x_ln, x_ln, x_ln, need_weights=False)
            x = x + self.dropp(x_res)
        if self.cross_attention:
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)
            x_res, _ = self.cross_mha(x_ln, enc, enc, need_weights=False)
            x = x + self.dropp(x_res)
        if self.use_mlp:
            x = self.mlp(x)
        return x
