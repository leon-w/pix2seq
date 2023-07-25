from datetime import datetime

from colorama import Fore, Style

import torch


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
                return red(f"T[x={self.o.numpy()}, {repr(self.o.dtype)}]")
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


class CallObserver:
    def __init__(self, obj):
        self.name = obj.__class__.__name__
        self.obj = obj

    def __call__(self, *args, **kwargs):
        p(f"{self.name}.__call__", *args, **kwargs)
        return self.obj(*args, **kwargs)


# [f"{k} -> {tuple(v.shape)}" for k, v in locals().items() if isinstance(v, torch.Tensor)]
