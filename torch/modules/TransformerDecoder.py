from TransformerDecoderLayer import TransformerDecoderLayer

import torch
import torch.nn as nn

# # TENSORFLOW IMPLEMENTATION
# class TransformerDecoder_tf(tf.keras.layers.Layer):
#   def __init__(self,
#                num_layers,
#                dim,
#                mlp_ratio,
#                num_heads,
#                drop_path=0.1,
#                drop_units=0.1,
#                drop_att=0.,
#                dim_x_att=None,
#                self_attention=True,
#                cross_attention=True,
#                use_mlp=True,
#                use_enc_ln=False,
#                use_ffn_ln=False,
#                ln_scale_shift=True,
#                **kwargs):
#     super(TransformerDecoder_tf, self).__init__(**kwargs)
#     self.num_layers = num_layers
#     self.dec_layers = [
#         TransformerDecoderLayer_tf(
#             dim,
#             mlp_ratio,
#             num_heads,
#             drop_path,
#             drop_units,
#             drop_att,
#             dim_x_att=dim_x_att,
#             self_attention=self_attention,
#             cross_attention=cross_attention,
#             use_mlp=use_mlp,
#             use_enc_ln=use_enc_ln,
#             use_ffn_ln=use_ffn_ln,
#             ln_scale_shift=ln_scale_shift,
#             name='transformer_decoder_layer' + suffix_id(i))
#         for i in range(num_layers)
#     ]

#   def call(self, x, enc, caches, mask_self, mask_cross, training):
#     """x in (bsz, seq, d), enc in (bsz, seq', d)."""
#     presents = []
#     for i in range(self.num_layers):
#       cache = None if caches is None else caches[i]
#       x, x_for_cache = self.dec_layers[i](
#           x, enc, cache, mask_self, mask_cross, training)
#       presents.append(x_for_cache)

#     return x, tf.stack(presents)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        mlp_ratio,
        num_heads,
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
        self.dec_layers = nn.ModuleList(
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

    def forward(self, x, enc, caches, mask_self, mask_cross, training):
        """x in (bsz, seq, d), enc in (bsz, seq', d)."""
        presents = []
        for i in range(self.num_layers):
            cache = None if caches is None else caches[i]
            x, x_for_cache = self.dec_layers[i](x, enc, cache, mask_self, mask_cross, training)
            presents.append(x_for_cache)

        return x, torch.stack(presents)
