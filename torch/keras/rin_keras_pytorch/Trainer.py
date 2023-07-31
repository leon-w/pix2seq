import math
from pathlib import Path

from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from torch_optimizer import Lamb
from tqdm import tqdm

import torch
import wandb
from torch.utils.data import DataLoader

from . import Rin, RinPipeline


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


# adapted from https://github.com/lucidrains/recurrent-interface-network-pytorch
class Trainer:
    def __init__(
        self,
        rin_model: Rin,
        scheduler,
        dataset,
        *,
        self_cond_rate=0.9,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        lr_warmup_steps=10000,
        train_num_steps=100000,
        betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=64,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        num_workers=2,
        run_name=None,
    ):
        super().__init__()

        wandb.init(project="rin", name=run_name)

        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision="fp16" if fp16 else "no")
        self.accelerator.native_amp = amp

        self.model = rin_model
        self.scheduler = scheduler
        self.pipeline = RinPipeline(rin_model, scheduler)

        self.self_cond_rate = self_cond_rate

        assert math.sqrt(num_samples).is_integer(), "num_samples must be a square number"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        self.ds = dataset
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = torch.optim.AdamW(rin_model.parameters(), lr=train_lr, betas=betas)
        # self.opt = Lamb(rin_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # lr scheduler

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.opt,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step + 1,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if self.accelerator.scaler is not None and data["scaler"] is not None:
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                batch_img, batch_class = next(self.dl)
                batch_class = torch.nn.functional.one_hot(batch_class, num_classes=10).float()

                timesteps = torch.randint(
                    low=0,
                    high=self.scheduler.config.num_train_timesteps,
                    size=(batch_img.size(0),),
                    device="cuda",
                    dtype=torch.long,
                )
                timesteps_normalized = timesteps / self.scheduler.config.num_train_timesteps

                noise = torch.randn_like(batch_img)
                noisy_image = self.scheduler.add_noise(batch_img, noise, timesteps)
                noisy_image = noisy_image.permute(0, 2, 3, 1)  # keras uses channels first

                latent_prev = None
                tape_prev = None
                with self.accelerator.autocast():
                    if torch.rand(1) < self.self_cond_rate:
                        with torch.no_grad():
                            _, latent_prev, tape_prev = self.model(
                                x=noisy_image, t=timesteps_normalized, cond=batch_class
                            )

                    pred, _, _ = self.model(
                        x=noisy_image,
                        t=timesteps_normalized,
                        cond=batch_class,
                        latent_prev=latent_prev,
                        tape_prev=tape_prev,
                    )
                    pred = pred.permute(0, 3, 1, 2)  # channels first to channels last
                    loss = torch.nn.functional.mse_loss(pred, noise)

                self.accelerator.backward(loss)

                pbar.set_description(f"loss: {loss.item():.4f}")

                self.accelerator.wait_for_everyone()

                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                        },
                        step=self.step,
                    )

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.model.eval()

                        samples = self.pipeline(batch_size=8 * 8, num_inference_steps=100)

                        sample_grid = rearrange(samples.cpu().numpy(), "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8)
                        sample_grid = (sample_grid + 1) / 2

                        wandb.log({"samples": wandb.Image(sample_grid)}, step=self.step)

                        self.save(milestone="latest")

                        self.model.train()

                self.step += 1
                pbar.update(1)

        self.accelerator.print("training complete")
