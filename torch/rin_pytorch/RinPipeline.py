from diffusers import DiffusionPipeline
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import randn_tensor

import torch

from .Rin import Rin


class RinPipeline(DiffusionPipeline):
    rin: Rin
    scheduler: DDPMScheduler

    def __init__(self, rin: Rin, scheduler: DDPMScheduler):
        super().__init__()

        self.set_progress_bar_config(desc="Denoising...", position=1, leave=False)

        self.register_modules(rin=rin, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=None, classes=None, num_inference_steps=50, clamp=True):
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        rin_device = next(self.rin.parameters()).device

        class_cond = self._get_class_cond(classes=classes, batch_size=batch_size).to(rin_device)
        output = randn_tensor((batch_size, *self.rin.image_shape), device=rin_device)
        latent_prev = None
        tape_prev = None

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output, latent_prev, tape_prev = self.rin(
                x=output,
                t=t / self.scheduler.config.num_train_timesteps,  # the model expects a value between 0 and 1
                cond=class_cond,
                latent_prev=latent_prev,
                tape_prev=tape_prev,
            )
            output = self.scheduler.step(model_output, t, output).prev_sample

        if clamp:
            output.clamp_(-1, 1)

        return output

    def _get_class_cond(self, classes=None, batch_size=None):
        num_classes = self.rin._cond_in_dim

        if classes is None:
            assert batch_size is not None
            classes = torch.randint(0, num_classes, (batch_size,))

        return torch.nn.functional.one_hot(classes, num_classes=num_classes).float()
