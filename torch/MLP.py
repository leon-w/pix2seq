from DropPath import DropPath
from FeedForwardLayer import FeedForwardLayer

import torch.nn as nn

# # TENSORFLOW IMPLEMENTATION
# class MLP_tf(tf.keras.layers.Layer):
#     def __init__(
#         self, num_layers, dim, mlp_ratio, drop_path=0.1, drop_units=0.0, use_ffn_ln=False, ln_scale_shift=True, **kwargs
#     ):
#         super(MLP_tf, self).__init__(**kwargs)
#         self.num_layers = num_layers
#         self.mlp_layers = [
#             FeedForwardLayer_tf(
#                 dim,
#                 dim * mlp_ratio,
#                 drop_units,
#                 use_ln=use_ffn_ln,
#                 ln_scale_shift=ln_scale_shift,
#                 name="ffn" + suffix_id(i),
#             )
#             for i in range(num_layers)
#         ]
#         self.layernorms = [
#             tf.keras.layers.LayerNormalization(
#                 epsilon=1e-6, center=ln_scale_shift, scale=ln_scale_shift, name="ffn/ln" + suffix_id(i)
#             )
#             for i in range(num_layers)
#         ]
#         self.dropp = DropPath_tf(drop_path)

#     def call(self, x, training, ret_list=False):
#         x_list = [x]
#         for i in range(self.num_layers):
#             x_residual = self.mlp_layers[i](self.layernorms[i](x), training)
#             x = x + self.dropp(x_residual, training)
#             x_list.append(x)
#         return (x, x_list) if ret_list else x


# PYTORCH IMPLEMENTATION
class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        mlp_ratio,
        drop_path=0.1,
        drop_units=0.0,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList(
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
        self.layernorms = nn.ModuleList(
            [
                nn.LayerNorm(
                    dim,
                    eps=1e-6,
                    elementwise_affine=ln_scale_shift,
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
