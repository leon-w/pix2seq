from einops import rearrange
from TapeDenoiser import TapeDenoiser

import torch.nn as nn

# class ImageTapeDenoiser_tf(TapeDenoiser_tf):
#     """Deal with image data of shape (bsz, h, w, c)."""
#     def __init__(
#         self,
#         num_layers,
#         latent_slots,
#         latent_dim,
#         latent_mlp_ratio,
#         latent_num_heads,
#         tape_dim,
#         tape_mlp_ratio,
#         rw_num_heads,
#         image_height,
#         image_width,
#         image_channels,
#         patch_size,
#         latent_pos_encoding="learned",
#         tape_pos_encoding="learned",
#         drop_path=0.0,
#         drop_units=0.1,
#         drop_att=0.0,
#         time_scaling=1e4,
#         self_cond="none",
#         **kwargs,
#     ):
#         self._n_rows = image_height // patch_size
#         self._n_cols = image_width // patch_size
#         self._num_tokens = self._n_rows * self._n_cols
#         self._patch_size = patch_size
#         self._output_dim = patch_size**2 * image_channels
#         super().__init__(
#             num_layers=num_layers,
#             latent_slots=latent_slots,
#             latent_dim=latent_dim,
#             latent_mlp_ratio=latent_mlp_ratio,
#             latent_num_heads=latent_num_heads,
#             tape_dim=tape_dim,
#             tape_mlp_ratio=tape_mlp_ratio,
#             rw_num_heads=rw_num_heads,
#             latent_pos_encoding=latent_pos_encoding,
#             tape_pos_encoding=tape_pos_encoding,
#             drop_path=drop_path,
#             drop_units=drop_units,
#             drop_att=drop_att,
#             time_scaling=time_scaling,
#             self_cond=self_cond,
#             **kwargs,
#         )

#         self.stem = tf.keras.layers.Conv2D(
#             filters=tape_dim, kernel_size=patch_size, strides=patch_size, padding="VALID", use_bias=True, name="stem"
#         )

#     def _x_to_tape(self, x, offset=0):
#         tokens = self.stem(x)
#         tokens = rearrange(tokens, "b h w d -> b (h w) d")
#         tape_pos_emb = rearrange(self.tape_pos_emb, "n d -> 1 n d")
#         if self._tape_pos_encoding in ["sin_cos_plus_learned"]:
#             tape_pos_emb += rearrange(self.tape_pos_emb_res, "n d -> 1 n d")
#         tokens = self.stem_ln(tokens) + tape_pos_emb
#         return tokens

#     def readout_tape(self, tape):
#         tokens = super().readout_tape(tape)
#         tokens = rearrange(
#             tokens,
#             "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
#             h=self._n_rows,
#             w=self._n_cols,
#             p1=self._patch_size,
#             p2=self._patch_size,
#         )
#         return tokens


class ImageTapeDenoiser(TapeDenoiser):
    """Deal with image data of shape (bsz, h, w, c)."""

    def __init__(
        self,
        num_layers,
        latent_slots,
        latent_dim,
        latent_mlp_ratio,
        latent_num_heads,
        tape_dim,
        tape_mlp_ratio,
        rw_num_heads,
        image_height,
        image_width,
        image_channels,
        patch_size,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.0,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1e4,
        self_cond="none",
        **kwargs,
    ):
        self._n_rows = image_height // patch_size
        self._n_cols = image_width // patch_size
        self._num_tokens = self._n_rows * self._n_cols
        self._patch_size = patch_size
        self._output_dim = patch_size**2 * image_channels
        super().__init__(
            num_layers=num_layers,
            latent_slots=latent_slots,
            latent_dim=latent_dim,
            latent_mlp_ratio=latent_mlp_ratio,
            latent_num_heads=latent_num_heads,
            tape_dim=tape_dim,
            tape_mlp_ratio=tape_mlp_ratio,
            rw_num_heads=rw_num_heads,
            latent_pos_encoding=latent_pos_encoding,
            tape_pos_encoding=tape_pos_encoding,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            time_scaling=time_scaling,
            self_cond=self_cond,
            **kwargs,
        )

        self.stem = nn.Conv2d(
            in_channels=image_channels,
            out_channels=tape_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def _x_to_tape(self, x, offset=0):
        tokens = self.stem(x)
        tokens = rearrange(tokens, "b c h w -> b (h w) c")
        tape_pos_emb = rearrange(self.tape_pos_emb, "n d -> 1 n d")
        if self._tape_pos_encoding in ["sin_cos_plus_learned"]:
            tape_pos_emb += rearrange(self.tape_pos_emb_res, "n d -> 1 n d")
        tokens = self.stem_ln(tokens) + tape_pos_emb
        return tokens

    def readout_tape(self, tape):
        tokens = super().readout_tape(tape)
        tokens = rearrange(
            tokens,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=self._n_rows,
            w=self._n_cols,
            p1=self._patch_size,
            p2=self._patch_size,
        )
        return tokens
