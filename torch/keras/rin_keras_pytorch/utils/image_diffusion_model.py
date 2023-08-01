from pathlib import Path

import torch.nn as nn
from ema_pytorch import EMA
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

import torch
import wandb

from .. import Rin
from . import diffusion_utils


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


class RinDiffusionModel(nn.Module):
    def __init__(
        self,
        rin: Rin,
        train_schedule: str,
        inference_schedule: str,
        pred_type: str,
        self_cond: str = "none",
        num_classes: int = 10,
        conditional: str = "class",
        self_cond_rate: float = 0.9,
        loss_type: str = "x",
    ):
        super().__init__()
        self._inference_schedule = inference_schedule
        self._pred_type = pred_type
        self._self_cond = self_cond
        self._num_classes = num_classes
        self._conditional = conditional
        self._self_cond_rate = self_cond_rate
        self._loss_type = loss_type

        self.scheduler = diffusion_utils.Scheduler(train_schedule)

        self.denoiser = rin
        # self.denoiser_ema = EMA(rin)

    def denoise(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = gamma.squeeze()
        assert gamma.ndim == 1
        denoiser = self.denoiser if training else self.denoiser
        output, latent, tape = denoiser(x, gamma, cond, latent_prev, tape_prev)
        return output, latent, tape

    @torch.no_grad()
    def sample(self, num_samples=100, iterations=100, method="ddim"):
        self.denoiser.eval()
        samples_shape = [num_samples, *self.denoiser.output_shape]
        if self._conditional == "class":
            # generate random classes
            cond = torch.randint(self._num_classes, [num_samples])
            cond = torch.nn.functional.one_hot(cond, self._num_classes).float().cuda()
        else:
            cond = None

        ts = torch.ones([num_samples] + [1, 1, 1])
        get_step = lambda t: ts * (1.0 - t / iterations)
        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(self._inference_schedule)

        samples = self.scheduler.sample_noise(samples_shape).cuda()
        noise_pred, data_pred = torch.zeros_like(samples).cuda(), torch.zeros_like(samples).cuda()
        x0_clip_fn = diffusion_utils.get_x0_clipping_function("-1,1")

        latent_prev = None
        tape_prev = None
        for t in tqdm(torch.arange(iterations, dtype=torch.float32), desc="sampling", leave=False, position=1):
            time_step = get_step(t)
            time_step_p = torch.max(get_step(t + 1), torch.tensor(0.0))
            gamma, gamma_prev = time_transform(time_step).cuda(), time_transform(time_step_p).cuda()

            pred_out, latent_prev, tape_prev = self.denoise(
                samples, gamma, cond, latent_prev, tape_prev, training=False
            )
            x0_eps = diffusion_utils.get_x0_eps(
                samples, gamma, pred_out, self._pred_type, x0_clip_fn, truncate_noise=True
            )
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]
            samples = self.scheduler.transition_step(
                samples=samples,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma,
                gamma_prev=gamma_prev,
                sampler_name=method,
            )

        samples = data_pred * 0.5 + 0.5  # convert -1,1 -> 0,1

        return samples

    def noise_denoise(self, images, labels, time_step=None, training=True):
        images = images * 2.0 - 1.0
        images_noised, noise, _, gamma = self.scheduler.add_noise(images, time_step=time_step)
        latent_prev = None
        tape_prev = None
        if self._self_cond != "none" and self._self_cond_rate > 0.0:
            bsz = images.size(0)
            mask = torch.rand(bsz) < self._self_cond_rate

            if torch.any(mask):
                _, latent_prev_out, tape_prev_out = self.denoise(
                    images_noised[mask], gamma[mask], labels[mask], training=training
                )

                latent_prev = torch.zeros((bsz, *latent_prev_out.shape[1:]), device=latent_prev_out.device)
                tape_prev = torch.zeros((bsz, *tape_prev_out.shape[1:]), device=tape_prev_out.device)

                latent_prev[mask] = latent_prev_out
                tape_prev[mask] = tape_prev_out

        denoise_out, _, _ = self.denoise(images_noised, gamma, labels, latent_prev, tape_prev, training=training)

        x0_clip_fn = diffusion_utils.get_x0_clipping_function("-1,1")
        pred_dict = diffusion_utils.get_x0_eps(
            images_noised, gamma, denoise_out, self._pred_type, x0_clip_fn, truncate_noise=False
        )
        return images, noise, images_noised, pred_dict

    def compute_loss(
        self, images: torch.Tensor, noise: torch.Tensor, pred_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self._loss_type == "x":
            loss = torch.mean(torch.square(images - pred_dict["data_pred"]))
        elif self._loss_type == "eps":
            loss = torch.mean(torch.square(noise - pred_dict["noise_pred"]))
        else:
            raise ValueError(f"Unknown pred_type {self._pred_type}")
        return loss

    def forward(self, images: torch.Tensor, labels: torch.Tensor, training=True) -> torch.Tensor:
        images, noise, _, pred_dict = self.noise_denoise(images, labels, training=training)
        loss = self.compute_loss(images, noise, pred_dict)
        return loss


class Trainer:
    def __init__(
        self,
        diffusion_model: RinDiffusionModel,
        dataset: Dataset,
        train_num_steps: int,
        lr: float,
        batch_size: int,
        sample_every: int,
        run_name: str = "debug",
        results_folder: str = "results",
    ):
        self.diffusion_model = diffusion_model
        self.train_num_steps = train_num_steps
        self.sample_every = sample_every

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.dataset = dataset
        self.dl = cycle(DataLoader(self.dataset, batch_size=batch_size))

        self.opt = torch.optim.Adam(self.diffusion_model.parameters(), lr=lr)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=train_num_steps)

        self.step = 0

        wandb.init(project="rin", name=run_name)

    def save(self, milestone):
        data = {
            "step": self.step + 1,
            "model": self.diffusion_model.denoiser.state_dict(),
            "opt": self.opt.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"))

        self.step = data["step"]
        self.diffusion_model.denoiser.load_state_dict(data["model"])
        self.opt.load_state_dict(data["opt"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                batch_img, batch_class = next(self.dl)
                batch_class = torch.nn.functional.one_hot(batch_class, num_classes=10).float()
                batch_img = batch_img.to("cuda")
                batch_class = batch_class.to("cuda")

                loss = self.diffusion_model(batch_img, batch_class)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                pbar.set_description(f"loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]}, step=self.step)

                self.step += 1
                pbar.update(1)

                if self.step % self.sample_every == 0:
                    samples = self.diffusion_model.sample(num_samples=64, iterations=400, method="ddim")
                    samples = make_grid(samples, nrow=8, normalize=True, range=(0, 1))
                    wandb.log({"samples": [wandb.Image(samples)]}, step=self.step)

                    self.save("latest")

                    del samples

                self.diffusion_model.denoiser.train()
