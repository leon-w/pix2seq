import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

from rin_pytorch import Rin, RinDiffusionModel
from rin_pytorch.utils.debug_utils import track
from rin_pytorch.utils.optimization_utils import build_torch_parameters_to_keras_names_mapping

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


def print_torch_params(model):
    keras_mapping = build_torch_parameters_to_keras_names_mapping(model)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        parts = name.split(".")

        if len(parts) > 1 and parts[-2] == "torch_params":
            assert id(param) in keras_mapping
            mapped_name = keras_mapping[id(param)]
            name = ".".join(parts[:-2] + [mapped_name])

        print(name, list(param.shape))


def print_tf_params():
    weight_pre = np.load("rin_cifar10_pretrained_weights.npy", allow_pickle=True).item()
    for name, param in weight_pre.items():
        name, _ = name.split(":")
        parts = name.split("/")
        name = ".".join(parts[1:])

        print(name, list(param.shape))


def load_weights(model):
    weight_pre = np.load("rin_cifar10_pretrained_weights.npy", allow_pickle=True).item()
    weights = list(weight_pre.values())

    trainable_params = list(p for p in model.parameters() if p.requires_grad)

    for i, param in enumerate(trainable_params):
        data = torch.from_numpy(weights[i]).cuda()
        param.data.copy_(data)


load_weights(rin)


# rin.eval()
# with torch.no_grad():
#     rng = np.random.default_rng(42)
#     x = torch.from_numpy(rng.random((1, 3, 32, 32), dtype=np.float32)).cuda()
#     t = torch.from_numpy(np.full((1,), 0.5, dtype=np.float32)).cuda()
#     cond = torch.from_numpy(rng.random((1, 10), dtype=np.float32)).cuda()

#     track("in", x=x, t=t, cond=cond)
#     output, latent_prev, tape_prev = rin(x, t, cond)
#     track("out-0", output=output, latent_prev=latent_prev, tape_prev=tape_prev)
#     output, latent_prev, tape_prev = rin(x, t, cond, latent_prev, tape_prev)
#     track("out-1", output=output, latent_prev=latent_prev, tape_prev=tape_prev)

diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])

diffusion_model.eval()

samples = diffusion_model.sample(64, 10)
sample_grid = rearrange(samples.detach().cpu().numpy(), "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8)
Image.fromarray((sample_grid * 255).astype("uint8")).save("samples_pt.png")
