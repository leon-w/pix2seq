from einops import rearrange
from LambdaModule import LambdaModule
from MLP import MLP
from pos_embedding import create_2d_pos_emb
from ScalarEmbedding import ScalarEmbedding
from TransformerDecoder import TransformerDecoder
from TransformerEncoder import TransformerEncoder

import torch
import torch.nn as nn


class TapeDenoiser(nn.Module):
    # TODO
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
        self.stem_ln = nn.LayerNorm(normalized_shape=tape_dim, eps=1e-6)
        self.time_emb = ScalarEmbedding(
            dim=(latent_dim if self._time_on_latent else cond_dim) // 4,
            scaling=time_scaling,
            expansion=4,
        )
        if cond_proj:
            # TODO: figure out input shapes
            self.cond_proj = nn.Linear(out_features=latent_dim if self._cond_on_latent else cond_dim)
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
            self.latent_prev_ln = nn.LayerNorm(normalized_shape=latent_dim, eps=1e-6)
            nn.init.zeros_(self.latent_prev_ln.weight)
        if self_cond in ["tape", "latent+tape"]:
            self.tape_prev_proj = MLP(
                num_layers=1,
                dim=tape_dim,
                mlp_ratio=tape_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.tape_prev_ln = nn.LayerNorm(normalized_shape=tape_dim, eps=1e-6)
            nn.init.zeros_(self.tape_prev_ln.weight)
        self.read_units = nn.ModuleList()
        self.read_cond_units = nn.ModuleList()
        self.write_units = nn.ModuleList()
        self.latent_processing_units = nn.ModuleList()
        for num_layers_per_readwrite in self._num_layers:
            self.read_units.append(
                TransformerDecoder(
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
            )
            if cond_decoupled_read:
                self.read_cond_units.append(
                    TransformerDecoder(
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
                )
            if num_layers_per_readwrite == 0:
                self.write_units.append(LambdaModule(lambda x: (x, None)))
                self.latent_processing_units.append(LambdaModule(lambda x: x))
            else:
                self.write_units.append(
                    TransformerDecoder(
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
                )
                self.latent_processing_units.append(
                    TransformerEncoder(
                        num_layers=num_layers_per_readwrite,
                        dim=latent_dim,
                        mlp_ratio=latent_mlp_ratio,
                        num_heads=latent_num_heads,
                        drop_path=drop_path,
                        drop_units=drop_units,
                        drop_att=drop_att,
                    )
                )
        self.output_ln = nn.LayerNorm(normalized_shape=self._output_dim, eps=1e-6)
        self.output_linear = nn.Linear(in_features=self._output_dim, out_features=self._output_dim)

    # done
    def make_latent_pos(
        self,
        latent_slots: int,
        latent_dim: int,
        latent_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if latent_pos_encoding in ["sin_cos_plus_learned"]:
            self.latent_pos_emb = create_2d_pos_emb(
                pos_encoding="sin_cos",
                n_rows=latent_slots,
                n_cols=1,
                dim=latent_dim,
                normalization_max=time_scaling,
            )
            self.latent_pos_emb_res = nn.Parameter(torch.empty(latent_slots, latent_dim))
            nn.init.zeros_(self.latent_pos_emb_res)
        elif latent_pos_encoding in ["learned", "sin_cos"]:
            self.latent_pos_emb = create_2d_pos_emb(
                pos_encoding=latent_pos_encoding,
                n_rows=latent_slots,
                n_cols=1,
                dim=latent_dim,
                normalization_max=time_scaling,
            )
        else:
            raise ValueError(f"Unknown latent_pos_encoding `{latent_pos_encoding}`")

    # done
    def make_tape_pos(
        self,
        tape_dim: int,
        tape_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if tape_pos_encoding in ["sin_cos_plus_learned"]:
            self.tape_pos_emb = create_2d_pos_emb(
                pos_encoding="sin_cos",
                n_rows=self._n_rows,
                n_cols=self._n_cols,
                dim=tape_dim,
                normalization_max=time_scaling,
            )
            self.tape_pos_emb_res = nn.Parameter(torch.empty(self._n_rows * self._n_cols, tape_dim))
            nn.init.zeros_(self.tape_pos_emb_res)
        elif tape_pos_encoding in ["learned", "sin_cos"]:
            self.tape_pos_emb = create_2d_pos_emb(
                pos_encoding=tape_pos_encoding,
                n_rows=self._n_rows,
                n_cols=self._n_cols,
                dim=tape_dim,
                normalization_max=time_scaling,
            )
        else:
            raise ValueError(f"Unknown tape_pos_encoding `{tape_pos_encoding}`")

    # done
    def initialize_cond(
        self,
        t: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if t is not None:
            t = self.time_emb(t, last_swish=False, normalize=True)
            t = rearrange(t, "b d -> b 1 d")
        if cond is not None:
            cond = self.cond_proj(cond)
            if cond.ndim == 2:
                cond = rearrange(cond, "b d -> b 1 d")
        return t, cond

    # done
    def initialize_tape(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        tape_prev: torch.Tensor | None,
    ):
        tape_r = None
        if not self._time_on_latent and time_emb is not None:
            tape_r = time_emb
        if not self._cond_on_latent and cond is not None:
            tape_r = self._concat_tokens(tape_r, cond)

        tape = self._x_to_tape(x)
        if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
            tape += self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        if self._cond_tape_writable and tape_r is not None:
            tape, tape_r = self._concat_tokens(tape, tape_r), None

        return tape, tape_r

    # done
    def initialize_latent(
        self,
        batch_size: int,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        latent = self.latent_pos_emb
        if self._latent_pos_encoding in ["sin_cos_plus_learned"]:
            latent += self.latent_pos_emb_res
        latent = latent.repeat(batch_size, 1, 1)
        if self._time_on_latent and time_emb is not None:
            latent = self._concat_tokens(latent, time_emb)
        if self._cond_on_latent and cond is not None:
            latent = self._concat_tokens(latent, cond)
        if self._self_cond in ["latent", "latent+tape"] and latent_prev is not None:
            latent += self.latent_prev_ln(self.latent_prev_proj(latent_prev))
        return latent

    # done
    def _concat_tokens(self, *tokens: torch.Tensor | None) -> torch.Tensor:
        # tokens in shape [..., n, d]
        return torch.cat([t for t in tokens if t is not None], -2)

    def compute(
        self,
        latent: torch.Tensor,
        tape: torch.Tensor,
        tape_r: torch.Tensor | None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(len(self._num_layers)):
            if self._cond_decoupled_read:
                latent = self.read_cond_units[i](latent, tape_r, None, None, None, training)[0]
                latent = self.read_units[i](latent, tape, None, None, None, training)[0]
            else:
                tape_merged = self._concat_tokens(tape, tape_r)
                latent = self.read_units[i](latent, tape_merged, None, None, None, training)[0]
            latent = self.latent_processing_units[i](latent, None, training)
            tape = self.write_units[i](tape, latent, None, None, None, training)[0]
        return latent, tape

    # done
    def readout_tape(self, tape: torch.Tensor) -> torch.Tensor:
        tokens = tape[:, : self._num_tokens]
        tokens = self.output_ln(tokens)
        tokens = self.output_linear(tokens)
        return tokens

    # done
    @property
    def latent_shape(self) -> list[int]:
        return [self._latent_slots, self._latent_dim]

    # done
    @property
    def tape_shape(self) -> list[int]:
        return [self._tape_slots, self._tape_dim]

    # done
    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
        t: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x[0] in (bs, h, w, c), t in (bs,), cond in (bs, s, d)."""
        if isinstance(x, tuple) or isinstance(x, list):
            x, latent_prev, tape_prev = x
            bs = x.shape[0]
        else:
            bs = x.shape[0]
            latent_prev = torch.zeros(bs, *self.latent_shape)
            tape_prev = torch.zeros(bs, *self.tape_shape)
        time_emb, cond = self.initialize_cond(t, cond)
        tape, tape_r = self.initialize_tape(x, time_emb, cond, tape_prev)
        latent = self.initialize_latent(bs, time_emb, cond, latent_prev)
        latent, tape = self.compute(latent, tape, tape_r)
        x = self.readout_tape(tape)
        return x, latent, tape[:, : self._tape_slots]
