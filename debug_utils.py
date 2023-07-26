from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from colorama import Fore, Style


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
        if isinstance(self.o, (tf.Tensor, tf.Variable)):
            if self.o.shape.ndims == 0:
                return red(f"T[x={self.o.numpy()}, {repr(self.o.dtype)}]")
            return red(f"T[{self.o.shape}, {repr(self.o.dtype)}]")

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


class CallObserver:
    def __init__(self, obj):
        self.name = obj.__class__.__name__
        self.obj = obj

    def __call__(self, *args, **kwargs):
        p(f"{self.name}.__call__", *args, **kwargs)
        return self.obj(*args, **kwargs)


def plot_dist(**kwargs):
    fig, axs = plt.subplots(1, len(kwargs), figsize=(len(kwargs) * 5, 5), sharex=True)
    names = []
    for ax, (k, v) in zip(axs, kwargs.items()):
        names.append(k)
        if isinstance(v, (tf.Tensor, tf.Variable)):
            v = v.numpy()

        ax.hist(v.flatten(), bins=100, alpha=0.5, label=k, density=True)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_yticks([])
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(k)

    fig.suptitle(", ".join(names))
    fig.savefig(f"TF_dist_{'_'.join(names)}.png", dpi=300, bbox_inches="tight")
