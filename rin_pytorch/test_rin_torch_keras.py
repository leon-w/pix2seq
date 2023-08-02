import os

os.environ["KERAS_BACKEND"] = "torch"

import keras_core as keras

keras.backend.set_image_data_format("channels_first")

import torchvision

import torch
from rin_pytorch import Rin, RinDiffusionModel, Trainer

rin = Rin(
    num_layers="2,2,2",
    latent_slots=128,
    latent_dim=512,
    latent_mlp_ratio=4,
    latent_num_heads=16,
    tape_dim=256,
    tape_mlp_ratio=2,
    rw_num_heads=8,
    image_height=32,
    image_width=32,
    image_channels=3,
    patch_size=2,
    latent_pos_encoding="learned",
    tape_pos_encoding="learned",
    drop_path=0.1,
    drop_units=0.1,
    drop_att=0.0,
    time_scaling=1e4,
    self_cond="latent",
    time_on_latent=True,
    cond_on_latent_n=1,
    cond_tape_writable=False,
    cond_dim=0,
    cond_proj=True,
    cond_decoupled_read=False,
    xattn_enc_ln=False,
).to("cuda")


with torch.no_grad():
    rin.eval()
    rin(
        x=torch.zeros(1, 3, 32, 32).to("cuda"),
        t=0.0,
        cond=torch.zeros(1, 10).to("cuda"),
    )
    rin.train()
