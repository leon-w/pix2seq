from rin_keras_pytorch import Rin

import torch

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

bs = 64
x = torch.randn((bs, 3, 32, 32)).to("cuda")
# t = torch.full((bs,), 0.5).to("cuda")
t = torch.rand((bs,)).to("cuda")

classes = torch.nn.functional.one_hot(torch.randint(0, 10, (bs,)).to("cuda"), num_classes=10).float()

output, latent_prev, tape_prev = rin(x, t, classes)

# t_emb, _ = rin.initialize_cond(t, None)
# t_emb = t_emb.detach().cpu().numpy()

# print(t_emb)

# debug_utils.plot_model_weights(rin)


# for name, param in rin.named_parameters():
#     print(name, param.shape)
