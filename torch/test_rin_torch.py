from modules.ImageTapeDenoiser import ImageTapeDenoiser
from modules.utils.debug_utils import p, plot_dist

import torch

rin = ImageTapeDenoiser(
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
    cond_in_dim=10,
    cond_proj=True,
    cond_decoupled_read=False,
    xattn_enc_ln=False,
).to("cuda")

x = torch.randn((8, 3, 32, 32)).to("cuda")
t = torch.randn((8,)).to("cuda")
classes = torch.nn.functional.one_hot(torch.randint(0, 10, (8,)).to("cuda"), num_classes=10).float()

# pass some data to build the model
output, latent_prev, tape_prev = rin(x, t, classes)

p(output_std=output.std(), output_mean=output.mean())

p(x=x, t=t, classes=classes, output=output)
plot_dist(x=x, output=output, latent_prev=latent_prev, tape_prev=tape_prev)

for name, param in rin.named_parameters():
    print(name, param.shape)
