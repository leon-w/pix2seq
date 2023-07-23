from DepthwiseConvBlock import DepthwiseConvBlock
from MLP import MLP
from ScalarEmbedding import ScalarEmbedding, add_vis_pos_emb
from TransformerDecoder import TransformerDecoder
from TransformerEncoder import TransformerEncoder

import torch
import torch.nn as nn


class TapeDenoiser(nn.Module):
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
        conv_kernel_size=0,
        conv_drop_units=0,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.0,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1e4,
        self_cond="none",
        time_on_latent=False,
        cond_on_latent_n=0,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
    ):
        super().__init__()
        self._num_layers = [int(i) for i in num_layers.split(",")]
        self._latent_slots = latent_slots
        self._time_on_latent = time_on_latent
        self._cond_on_latent = cond_on_latent_n > 0
        if self._time_on_latent:  # replace 1 latent with time emb.
            latent_slots -= 1
        latent_slots -= cond_on_latent_n
        self._latent_dim = latent_dim
        self._tape_slots = self._num_tokens
        self._tape_dim = tape_dim
        self._cond_dim = cond_dim = cond_dim if cond_dim > 0 else tape_dim
        self._latent_pos_encoding = latent_pos_encoding
        self._tape_pos_encoding = tape_pos_encoding
        assert self_cond in ("none", "latent", "tape", "latent+tape")
        self._self_cond = self_cond
        self._cond_tape_writable = cond_tape_writable
        self._cond_decoupled_read = cond_decoupled_read
        time_scaling = torch.tensor(time_scaling, dtype=torch.float32)
        self.stem_ln = nn.LayerNorm(normalized_shape=latent_dim, eps=1e-6, elementwise_affine=True)
        self.time_emb = ScalarEmbedding(
            dim=(latent_dim if self._time_on_latent else cond_dim) // 4,
            scaling=time_scaling,
            expansion=4,
        )
        if cond_proj:
            self.cond_proj = nn.Linear(
                in_features=latent_dim if self._cond_on_latent else cond_dim,
                out_features=latent_dim,
            )
        else:
            self.cond_proj = nn.Identity()

        self.make_latent_pos(latent_slots, latent_dim, latent_pos_encoding, time_scaling)
        self.make_tape_pos(tape_dim, tape_pos_encoding, time_scaling)

        if self_cond in ["latent", "latent+tape"]:
            self.latent_prev_proj = MLP(
                num_layers=1,
                dim=latent_dim,
                mlp_ratio=latent_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.latent_prev_ln = nn.LayerNorm(normalized_shape=latent_dim, eps=1e-6, elementwise_affine=True)
        if self_cond in ["tape", "latent+tape"]:
            self.tape_prev_proj = MLP(
                num_layers=1,
                dim=tape_dim,
                mlp_ratio=tape_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.tape_prev_ln = nn.LayerNorm(normalized_shape=tape_dim, eps=1e-6, elementwise_affine=True)
        self.read_units = {}
        self.read_cond_units = {}
        self.write_units = {}
        self.conv_units = {}
        self.latent_processing_units = {}
        for i, num_layers_per_readwrite in enumerate(self._num_layers):
            self.read_units[str(i)] = TransformerDecoder(
                num_layers=1,
                dim=latent_dim,
                mlp_ratio=latent_mlp_ratio,
                num_heads=rw_num_heads,
                drop_path=0.0,
                drop_units=0.0,
                drop_att=0.0,
                dim_x_att=min(tape_dim, latent_dim),
                self_attention=False,
                cross_attention=True,
                use_mlp=True,
                use_enc_ln=xattn_enc_ln,
            )
            if cond_decoupled_read:
                self.read_cond_units[str(i)] = TransformerDecoder(
                    num_layers=1,
                    dim=latent_dim,
                    mlp_ratio=latent_mlp_ratio,
                    num_heads=rw_num_heads,
                    drop_path=0.0,
                    drop_units=0.0,
                    drop_att=0.0,
                    dim_x_att=min(cond_dim, latent_dim),
                    self_attention=False,
                    cross_attention=True,
                    use_mlp=True,
                    use_enc_ln=xattn_enc_ln,
                )
            if num_layers_per_readwrite == 0:
                self.write_units[str(i)] = lambda x, *args, **kwargs: (x, None)
                self.conv_units[str(i)] = lambda x, *args, **kwargs: x
                self.latent_processing_units[str(i)] = lambda x, *args, **kwargs: x
            else:
                self.write_units[str(i)] = TransformerDecoder(
                    num_layers=1,
                    dim=tape_dim,
                    mlp_ratio=tape_mlp_ratio,
                    num_heads=rw_num_heads,
                    drop_path=0.0,
                    drop_units=0.0,
                    drop_att=0.0,
                    dim_x_att=min(tape_dim, latent_dim),
                    self_attention=False,
                    cross_attention=True,
                    use_mlp=True if tape_mlp_ratio > 0 else False,
                    use_enc_ln=xattn_enc_ln,
                )
                if conv_kernel_size == 0:
                    self.conv_units[str(i)] = lambda x, *args, **kwargs: x
                else:
                    self.conv_units[str(i)] = DepthwiseConvBlock(
                        tape_dim, kernel_size=conv_kernel_size, dropout_rate=conv_drop_units
                    )
                self.latent_processing_units[str(i)] = TransformerEncoder(
                    num_layers=num_layers_per_readwrite,
                    dim=latent_dim,
                    mlp_ratio=latent_mlp_ratio,
                    num_heads=latent_num_heads,
                    drop_path=drop_path,
                    drop_units=drop_units,
                    drop_att=drop_att,
                )
        self.output_ln = nn.LayerNorm(normalized_shape=self._output_dim, eps=1e-6, elementwise_affine=True)
        self.output_linear = nn.Linear(in_features=self._output_dim, out_features=self._output_dim, bias=True)

    def make_latent_pos(self, latent_slots, latent_dim, latent_pos_encoding, time_scaling):
        if latent_pos_encoding in ["sin_cos_plus_learned"]:
            self.latent_pos_emb = add_vis_pos_emb(
                self,
                "sin_cos",
                latent_slots,
                1,
                latent_dim,
                name_prefix=f"{self.name}/latent_pos_emb/kernel",
                return_only=True,
                normalization_max=time_scaling,
            )
            self.latent_pos_emb_res = self.add_weight(
                shape=(latent_slots, latent_dim), initializer="zeros", name=f"{self.name}/latent_pos_emb_res/kernel"
            )
        elif latent_pos_encoding in ["learned", "sin_cos"]:
            self.latent_pos_emb = add_vis_pos_emb(
                self,
                latent_pos_encoding,
                latent_slots,
                1,
                latent_dim,
                return_only=True,
                normalization_max=time_scaling,
            )
        else:
            raise ValueError(f"Unknown latent_pos_encoding {latent_pos_encoding}")

    def make_tape_pos(self, tape_dim, tape_pos_encoding, time_scaling):
        if tape_pos_encoding in ["sin_cos_plus_learned"]:
            self.tape_pos_emb = add_vis_pos_emb(
                self,
                "sin_cos",
                self._n_rows,
                self._n_cols,
                tape_dim,
                return_only=True,
                normalization_max=time_scaling,
            )
            self.tape_pos_emb_res = self.add_weight(
                shape=(self._n_rows * self._n_cols, tape_dim),
                initializer="zeros",
                name=f"{self.name}/tape_pos_emb_res/kernel",
            )
        elif tape_pos_encoding in ["learned", "sin_cos"]:
            self.tape_pos_emb = add_vis_pos_emb(
                self,
                tape_pos_encoding,
                self._n_rows,
                self._n_cols,
                tape_dim,
                return_only=True,
                normalization_max=time_scaling,
            )
        else:
            raise ValueError(f"Unknown tape_pos_encoding {tape_pos_encoding}")

    def initialize_cond(self, t, cond, training):
        if t is not None:
            t = self.time_emb(t, last_swish=False, normalize=True).unsqueeze(1)
        if cond is not None:
            cond = self.cond_proj(cond)
            if cond.ndim == 2:
                cond = cond.unsqueeze(1)
        return t, cond

    def initialize_tape(self, x, time_emb, cond, tape_prev, offset=0):
        tape_r = None
        if not self._time_on_latent and time_emb is not None:
            tape_r = time_emb
        if not self._cond_on_latent and cond is not None:
            tape_r = cond if tape_r is None else torch.cat([tape_r, cond], 1)
        tape = self._x_to_tape(x, offset)  # (bsz, n, d)

        if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
            tape += self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        if self._cond_tape_writable and tape_r is not None:
            tape, tape_r = torch.cat([tape, tape_r], 1), None
        return tape, tape_r

    def initialize_latent(self, batch_size, time_emb, cond, latent_prev):
        latent = self.latent_pos_emb.unsqueeze(0)
        if self._latent_pos_encoding in ["sin_cos_plus_learned"]:
            latent += self.latent_pos_emb_res.unsqueeze(0)
        latent = latent.repeat(batch_size, 1, 1)
        if self._time_on_latent and time_emb is not None:
            latent = torch.cat([latent, time_emb], 1)
        if self._cond_on_latent and cond is not None:
            latent = torch.cat([latent, cond], 1)
        if self._self_cond in ["latent", "latent+tape"]:
            latent += self.latent_prev_ln(self.latent_prev_proj(latent_prev))
        return latent

    def _merge_tape(self, tape_writable, tape_readonly):
        tape_merged = tape_writable if tape_readonly is None else (torch.cat([tape_writable, tape_readonly], 1))
        return tape_merged

    def compute(self, latent, tape, tape_r, training):
        for i in range(len(self._num_layers)):
            if self._cond_decoupled_read:
                latent = self.read_cond_units[str(i)](latent, tape_r, None, None, None, training)[0]
                latent = self.read_units[str(i)](latent, tape, None, None, None, training)[0]
            else:
                tape_merged = self._merge_tape(tape, tape_r)
                latent = self.read_units[str(i)](latent, tape_merged, None, None, None, training)[0]
            latent = self.latent_processing_units[str(i)](latent, None, training)
            tape = self.write_units[str(i)](tape, latent, None, None, None, training)[0]
            tape = self.conv_units[str(i)](tape, training, size=self._num_tokens)
        return latent, tape

    def readout_tape(self, tape):
        tokens = self.output_linear(self.output_ln(tape[:, : self._num_tokens]))
        return tokens

    @property
    def hidden_shapes(self):
        latent_shape = [self._latent_slots, self._latent_dim]
        tape_shape = [self._tape_slots, self._tape_dim]
        return latent_shape, tape_shape

    def forward(self, x, t, cond, training):
        """x[0] in (bsz, h, w, c), t in (bsz, m), cond in (bsz, s, d)."""
        if isinstance(x, tuple) or isinstance(x, list):
            x, latent_prev, tape_prev = x
            bsz = x.shape[0]
        else:
            bsz = x.shape[0]
            latent_prev = torch.zeros([bsz] + self.hidden_shapes[0])
            tape_prev = torch.zeros([bsz] + self.hidden_shapes[1])
        time_emb, cond = self.initialize_cond(t, cond, training)
        tape, tape_r = self.initialize_tape(x, time_emb, cond, tape_prev)
        latent = self.initialize_latent(bsz, time_emb, cond, latent_prev)
        latent, tape = self.compute(latent, tape, tape_r, training)
        x = self.readout_tape(tape)
        return x, latent, tape[:, : self._tape_slots]
