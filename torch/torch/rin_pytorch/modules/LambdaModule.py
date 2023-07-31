from typing import Any, Callable

import torch.nn as nn


class LambdaModule(nn.Module):
    def __init__(self, lambd: Callable, ignore_args=True, ignore_kwargs=True):
        super().__init__()
        self.lambd = lambd
        self.no_args = ignore_args
        self.no_kwargs = ignore_kwargs

    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        if self.no_args and self.no_kwargs:
            return self.lambd(x)
        elif self.no_args:
            return self.lambd(x, **kwargs)
        elif self.no_kwargs:
            return self.lambd(x, *args)
        else:
            return self.lambd(x, *args, **kwargs)
