import os

os.environ["KERAS_BACKEND"] = "torch"

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
rin.load_weights_numpy("../rin_cifar10_fresh0_weights.npy")

# diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])
# diffusion_model.eval()

# samples = diffusion_model.sample(64, 100, "ddim", seed=42)
# track("sample_grid", samples=samples)
# sample_grid = rearrange(samples.detach().cpu().numpy(), "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8)
# Image.fromarray((sample_grid * 255).clip(0, 255).astype("uint8")).save("samples_pt.png")
