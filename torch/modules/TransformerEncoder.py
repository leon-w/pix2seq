from TransformerEncoderLayer import TransformerEncoderLayer

import torch.nn as nn

# # TENSORFLOW IMPLEMENTATION
# class TransformerEncoder_tf(tf.keras.layers.Layer):
#   def __init__(self,
#                num_layers,
#                dim,
#                mlp_ratio,
#                num_heads,
#                drop_path=0.1,
#                drop_units=0.1,
#                drop_att=0.,
#                self_attention=True,
#                use_ffn_ln=False,
#                ln_scale_shift=True,
#                **kwargs):
#     super(TransformerEncoder_tf, self).__init__(**kwargs)
#     self.num_layers = num_layers
#     self.enc_layers = [
#         TransformerEncoderLayer_tf(
#             dim,
#             mlp_ratio,
#             num_heads,
#             drop_path,
#             drop_units,
#             drop_att,
#             self_attention=self_attention,
#             use_ffn_ln=use_ffn_ln,
#             ln_scale_shift=ln_scale_shift,
#             name='transformer_encoder' + suffix_id(i))
#         for i in range(num_layers)
#     ]

#   def call(self, x, mask, training, ret_list=False):
#     x_list = [x]
#     for i in range(self.num_layers):
#       x = self.enc_layers[i](x, mask, training)
#       x_list.append(x)
#     return (x, x_list) if ret_list else x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        mlp_ratio,
        num_heads,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        self_attention=True,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.enc_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(  # pylint: disable=g-complex-comprehension
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

    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x
