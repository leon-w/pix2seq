from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

import torch
import wandb

from .RinDiffusionModel import RinDiffusionModel


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer:
    def __init__(
        self,
        diffusion_model: RinDiffusionModel,
        dataset: Dataset,
        train_num_steps: int,
        lr: float,
        batch_size: int,
        sample_every: int,
        run_name: str | None = None,
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

        wandb.init(project="rin", name=run_name, mode="disabled" if run_name is None else "online")

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
        self.diffusion_model.denoiser.train()

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

                logs = {
                    "loss": loss.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }

                pbar.set_postfix(logs)
                wandb.log(logs, step=self.step)

                self.step += 1
                pbar.update(1)
                self.lr_scheduler.step()

                if self.step % self.sample_every == 0:
                    samples = self.diffusion_model.sample(num_samples=64, iterations=400, method="ddim")

                    samples = make_grid(samples, nrow=8, normalize=True, range=(0, 1))
                    wandb.log({"samples": [wandb.Image(samples)]}, step=self.step)

                    self.save("latest")

                    del samples
