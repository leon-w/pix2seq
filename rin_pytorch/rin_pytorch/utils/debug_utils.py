import math
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore, Style
from einops import rearrange
from PIL import Image


def red(s):
    return f"{Fore.LIGHTRED_EX}{s}{Style.RESET_ALL}"


def yellow(s):
    return f"{Fore.YELLOW}{s}{Style.RESET_ALL}"


def green(s):
    return f"{Fore.GREEN}{s}{Style.RESET_ALL}"


class FormatObject:
    def __init__(self, o):
        self.o = o

    def __repr__(self):
        if isinstance(self.o, torch.Tensor):
            if self.o.ndim == 0:
                return red(f"T[x={self.o.detach().cpu().numpy()}, {repr(self.o.dtype)}]")
            return red(f"T[{tuple(self.o.shape)}, {repr(self.o.dtype)}]")

        if isinstance(self.o, tuple):
            return repr(tuple(FormatObject(x) for x in self.o))

        if isinstance(self.o, list):
            return repr([FormatObject(x) for x in self.o])

        return red(self.o)


def p(*args, **kwargs):
    time = green(f"[{datetime.now().strftime('%H:%M:%S')}]")

    items = []

    for arg in args:
        items.append(FormatObject(arg))

    for k, v in kwargs.items():
        items.append(yellow(f"{k}=") + repr(FormatObject(v)))

    print(time, *items)


track_counts = defaultdict(int)


def track(msg, **kwargs):
    for k, v in kwargs.items():
        if v is None:
            v_str = "None"
            p("WARNING", f"{k} is None")
        elif isinstance(v, torch.Tensor):
            t_np = v.detach().cpu().numpy()
            v_str = f"Tensor[{t_np.shape}, {t_np.mean():.6f} | {t_np.std():.6f} | {t_np.min():.6f} | {t_np.max():.6f}]"

            tensor_id = f"{msg}-{k}"

            index = track_counts[tensor_id]

            # save tensor
            np.save(f"tensors/PT-{msg}-{index}-{k}.npy", t_np)

            track_counts[tensor_id] += 1
        else:
            v_str = repr(v)
        # p(msg, f"{k}={v_str}")


class CallObserver(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.name = module.__class__.__name__
        self.module = module

    def __call__(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        p(f"{self.name}.__call__", *args, **kwargs, _="--->", output=output)
        return output


def plot_dist(**kwargs):
    fig, axs = plt.subplots(1, len(kwargs), figsize=(len(kwargs) * 5, 5), sharex=True)
    names = []
    for ax, (k, v) in zip(axs, kwargs.items()):
        names.append(k)
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()

        ax.hist(v.flatten(), bins=100, alpha=0.5, label=k, density=True)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks([])
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(k)

    fig.suptitle(", ".join(names))
    fig.savefig(f"TORCH_dist_{'_'.join(names)}.png", dpi=300, bbox_inches="tight")


class WeightTracker:
    def __init__(self, module, enable=True):
        self.module = module
        self.enable = enable
        self.weights = defaultdict(list)

        self.snapshot()

    def snapshot(self):
        if not self.enable:
            return

        for name, param in self.module.named_parameters():
            param = param.detach().cpu().numpy().copy()

            self.weights[name].append(param)

            while len(self.weights[name]) > 2:
                self.weights[name].pop(0)

        for name, values in self.weights.items():
            if len(values) != 2:
                continue
            last, prev = values
            diff = np.abs(last - prev).mean()

            p(f"{name} diff", diff)


def plot_model_weights(model):
    params = {}
    for name, param in model.named_parameters():
        param = param.detach().cpu().numpy().copy()
        params[name] = param

    w = math.ceil(math.sqrt(len(params)))
    h = math.ceil(len(params) / w)

    fig, axs = plt.subplots(h, w, figsize=(w * 5, h * 5), sharex=True)
    axs = axs.flatten()

    for ax, (name, param) in zip(axs, params.items()):
        ax.hist(param.flatten(), bins=100, alpha=0.5, label=name, density=True)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks([])
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(name)

    fig.suptitle("Model weights")
    fig.savefig(f"TORCH_model_weights.png", dpi=300, bbox_inches="tight")


# [f"{k} -> {tuple(v.shape)}" for k, v in locals().items() if isinstance(v, torch.Tensor)]


def save_image_grid(images, filename):
    n = math.sqrt(images.shape[0])
    assert n.is_integer()

    sample_grid = rearrange(images.detach().cpu().numpy(), "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n))
    Image.fromarray((sample_grid * 255).astype("uint8")).save(filename)
