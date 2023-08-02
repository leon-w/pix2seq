import os

os.environ["KERAS_BACKEND"] = "torch"

import torchvision
from rin_keras_pytorch import Rin, RinDiffusionModel, Trainer

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
).cuda()

rin.pass_dummy_data(num_classes=10)

diffusion_model = RinDiffusionModel(
    rin=rin,
    train_schedule="sigmoid@-3,3,0.9",
    inference_schedule="cosine",
    pred_type="eps",
    self_cond="latent",
    loss_type="eps",
)


# dataset
dataset = torchvision.datasets.CIFAR10(
    "/storage/slurm/wiemers/datasets/cifar10",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

trainer = Trainer(
    diffusion_model,
    dataset,
    train_num_steps=100_000,
    batch_size=256,
    sample_every=1000,
    lr=1e-4,
    results_folder="results/lamb-1e-4",
    run_name="lamb-1e-4",
)
trainer.train()
