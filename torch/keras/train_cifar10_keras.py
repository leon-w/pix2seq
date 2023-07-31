import os

os.environ["KERAS_BACKEND"] = "torch"

import torchvision
from diffusers.schedulers import DDPMScheduler
from rin_keras_pytorch import Rin, Trainer

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


# noise scheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="sigmoid")

# dataset
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x * 2) - 1),
    ]
)
dataset = torchvision.datasets.CIFAR10(
    "/storage/slurm/wiemers/datasets/cifar10",
    train=True,
    download=True,
    transform=image_transform,
)


# trainer

trainer = Trainer(
    rin_model=rin,
    scheduler=noise_scheduler,
    dataset=dataset,
    train_batch_size=256,
    self_cond_rate=0.9,
    gradient_accumulate_every=1,
    train_lr=3e-3,
    lr_warmup_steps=10,
    train_num_steps=100000,
    betas=(0.9, 0.99),
    save_and_sample_every=1000,
    num_samples=64,
    results_folder="./results",
    amp=False,
    fp16=False,
    split_batches=True,
    num_workers=1,
    run_name="rin-cifar10-debug",
)

trainer.train()
