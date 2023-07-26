from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch_optimizer as optim
import torchvision
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from modules.ImageTapeDenoiser import ImageTapeDenoiser
from torchvision.utils import save_image
from tqdm import tqdm

import torch
import wandb
from torch.rin_pytorch.RinPipeline import RinPipeline
from torch.utils.data import DataLoader

wandb.init(project="rin", name="pytorch-rin-cifar10-debug", mode="disabled")


@dataclass
class TrainConfig:
    train_steps: int = 75_000
    batch_size: int = 512
    checkpoint_steps: int = 500


config = TrainConfig()

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


# noise scheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="sigmoid")

# pipeline
pipeline = RinPipeline(rin, noise_scheduler)

# # get number of parameters
# num_params = sum(p.numel() for p in rin.parameters() if p.requires_grad)
# print(f"Number of parameters: {num_params:,}")

# dataset and dataloader
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
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
dataloader = cycle(dataloader)
dataloader_iter = iter(dataloader)

# optimizer
optimizer = torch.optim.Adam(rin.parameters(), lr=1e-3)

# lr scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=10000,
    num_training_steps=config.train_steps,
)


def train():
    rin.train()
    step = 0
    progress_bar = tqdm(total=config.train_steps, desc="Training")
    while step < config.train_steps:
        batch_img, batch_class = next(dataloader_iter)
        batch_class = torch.nn.functional.one_hot(batch_class, num_classes=10).float()
        batch_img = batch_img.to("cuda")
        batch_class = batch_class.to("cuda")

        timesteps = torch.randint(
            low=0,
            high=noise_scheduler.config.num_train_timesteps,
            size=(config.batch_size,),
            device="cuda",
            dtype=torch.long,
        )

        noise = torch.randn_like(batch_img)
        noisy_image = noise_scheduler.add_noise(batch_img, noise, timesteps)

        save_image((noisy_image.cpu() + 1) / 2, "samples_tf.png", nrow=16)

        latent_prev = None
        tape_prev = None

        timesteps_model = timesteps / noise_scheduler.config.num_train_timesteps

        if torch.rand(1) < 0.9:
            with torch.no_grad():
                _, latent_prev, tape_prev = rin(
                    x=noisy_image.detach(), t=timesteps_model.detach(), cond=batch_class.detach()
                )

        pred, _, _ = rin(
            x=noisy_image.detach(),
            t=timesteps_model.detach(),
            cond=batch_class.detach(),
            latent_prev=latent_prev,
            tape_prev=tape_prev,
        )

        # break

        loss = torch.nn.functional.mse_loss(pred, noise)

        wandb.log(
            {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": step,
            },
            step=step,
        )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % config.checkpoint_steps == 0:
            rin.eval()

            samples = pipeline(batch_size=8 * 8, num_inference_steps=100)

            torch.save(rin.state_dict(), "rin_latest.pt")

            sample_grid = rearrange(samples.cpu().numpy(), "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8)
            sample_grid = (sample_grid + 1) / 2

            if np.isnan(sample_grid).any():
                print(sample_grid)
                print("NAN encountered!!")
                exit()

            wandb.log({"samples": wandb.Image(sample_grid)}, step=step)

            rin.train()

        step += 1
        progress_bar.update(1)


train()
