import torch.nn as nn

# # TENSORFLOW IMPLEMENTATION
# class FeedForwardLayer_tf(tf.keras.layers.Layer):
#     def __init__(self, dim_att, dim_mlp, drop_units=0.1, use_ln=False, ln_scale_shift=False, **kwargs):
#         super(FeedForwardLayer_tf, self).__init__(**kwargs)
#         self.dense1 = tf.keras.layers.Dense(dim_mlp, activation=tf.nn.gelu, name="dense1")
#         self.dropout = tf.keras.layers.Dropout(drop_units)
#         self.dense2 = tf.keras.layers.Dense(dim_att, name="dense2")
#         if use_ln:
#             self.ln = tf.keras.layers.LayerNormalization(
#                 epsilon=1e-6, center=ln_scale_shift, scale=ln_scale_shift, name="mlp_ln"
#             )
#         else:
#             self.ln = lambda x: x

#     def call(self, x, training):
#         return self.dense2(self.dropout(self.ln(self.dense1(x)), training=training))


# PYTORCH IMPLEMENTATION
class FeedForwardLayer(nn.Module):
    def __init__(self, dim_att, dim_mlp, drop_units=0.1, use_ln=False, ln_scale_shift=False):
        super(FeedForwardLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_att, dim_mlp),
            nn.GELU(),
            nn.LayerNorm(dim_mlp, elementwise_affine=ln_scale_shift) if use_ln else nn.Identity(),
            nn.Dropout(drop_units),
            nn.Linear(dim_mlp, dim_att),
        )

    def forward(self, x):
        return self.model(x)
