from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler as get_lr_scheduler
from ema_pytorch import EMA
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb

from .RinDiffusionModel import RinDiffusionModel
from .utils.optimization_utils import (
    build_torch_parameters_to_keras_names_mapping,
    disable_weight_decay_for,
    get_optimizer,
)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer:
    def __init__(
        self,
        diffusion_model: RinDiffusionModel,
        ema_diffusion_model: RinDiffusionModel,  # since RinDiffusionModel can't be copied, we need a second one for EMA
        dataset: Dataset,
        train_num_steps: int,
        train_batch_size=256,
        split_batches=True,
        fp16=False,
        amp=False,
        lr_scheduler_name="cosine",
        lr=1e-4,
        lr_warmup_steps=1000,
        optimizer_name="lamb",
        optimizer_exclude_weight_decay=["bias", "beta", "gamma"],
        optimizer_kwargs=dict(weight_decay=1e-2),
        clip_grad_norm=1.0,
        sample_every=1000,
        num_dl_workers=2,
        checkpoint_folder="results",
        run_name="rin",
        log_to_wandb=True,
    ):
        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision="fp16" if fp16 else "no")
        self.accelerator.native_amp = amp

        self.diffusion_model = diffusion_model

        self.train_num_steps = train_num_steps
        self.clip_grad_norm = clip_grad_norm
        self.sample_every = sample_every

        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_dl_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.optimizer = get_optimizer(
            optimizer_name,
            disable_weight_decay_for(
                self.diffusion_model.parameters(),
                optimizer_exclude_weight_decay,
                build_torch_parameters_to_keras_names_mapping(self.diffusion_model),
            ),
            lr=lr,
            **optimizer_kwargs,
        )

        self.lr_scheduler = get_lr_scheduler(
            lr_scheduler_name,
            self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_num_steps,
        )

        if self.accelerator.is_main_process:
            self.ema_diffusion_model = EMA(diffusion_model, ema_model=ema_diffusion_model)

            self.checkpoint_folder = Path(checkpoint_folder)
            self.checkpoint_folder.mkdir(exist_ok=True, parents=True)

        self.step = 0

        self.diffusion_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.lr_scheduler
        )

        wandb.init(project="rin", name=run_name, mode="online" if log_to_wandb else "disabled")

    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "opt": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "ema": self.ema_diffusion_model.state_dict(),
        }

        torch.save(data, self.checkpoint_folder / f"model-{milestone}.pt")

    def load(self, milestone):
        data = torch.load(self.checkpoint_folder / f"model-{milestone}.pt")

        self.step = data["step"]

        diffusion_model = self.accelerator.unwrap_model(self.diffusion_model)
        diffusion_model.load_state_dict(data["model"])

        self.optimizer.load_state_dict(data["opt"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

        if self.accelerator.is_main_process:
            self.ema_diffusion_model.load_state_dict(data["ema"])

    def train(self):
        self.diffusion_model.train()

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                batch_img, batch_class = next(self.dl)
                batch_class = torch.nn.functional.one_hot(batch_class, num_classes=10).float()

                self.optimizer.zero_grad()

                loss = self.diffusion_model(batch_img, batch_class)

                self.accelerator.backward(loss)

                # self.accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                logs = {
                    "loss": loss.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }

                pbar.set_postfix(logs)
                wandb.log(logs, step=self.step)

                self.step += 1
                self.lr_scheduler.step()
                pbar.update(1)

                if self.accelerator.is_main_process:
                    self.ema_diffusion_model.update()

                    if self.step % self.sample_every == 0:
                        samples = self.diffusion_model.sample(num_samples=64, iterations=400, method="ddim")
                        samples = make_grid(samples, nrow=8, normalize=True, range=(0, 1))
                        wandb.log({"samples": [wandb.Image(samples)]}, step=self.step)

                        self.save("latest")

                        del samples
