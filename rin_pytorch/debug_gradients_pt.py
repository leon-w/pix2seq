import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

from rin_pytorch import Rin, RinDiffusionModel
from rin_pytorch.utils.debug_utils import track

config = dict(
    rin=dict(
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
        time_scaling=1000,
        self_cond="latent",
        time_on_latent=True,
        cond_on_latent_n=1,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
    ),
    diffusion=dict(
        train_schedule="sigmoid@-3,3,0.9",
        inference_schedule="cosine",
        pred_type="eps",
        self_cond="latent",
        loss_type="eps",
    ),
)


rin = Rin(**config["rin"]).cuda()
rin.pass_dummy_data(num_classes=10)


def load_weights(model):
    weight_pre = np.load("rin_cifar10_pretrained_weights.npy", allow_pickle=True).item()
    weights = list(weight_pre.values())

    trainable_params = list(p for p in model.parameters() if p.requires_grad)

    for i, param in enumerate(trainable_params):
        data = torch.from_numpy(weights[i]).cuda()
        param.data.copy_(data)


load_weights(rin)


# diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])
# diffusion_model.train()

rin.eval()

rng = np.random.default_rng(42)
x = torch.from_numpy(rng.random((1, 3, 32, 32), dtype=np.float32)).cuda()
t = torch.from_numpy(np.full((1,), 0.5, dtype=np.float32)).cuda()
cond = torch.from_numpy(rng.random((1, 10), dtype=np.float32)).cuda()

output, latent, tape = rin(x, t, cond)
loss = torch.square(output - x).mean()
loss.backward()

print(loss.item())

for g in rin.parameters():
    if g.requires_grad:
        track("g", g=g.grad)
