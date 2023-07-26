import math

import torch


def initialize_variable_variance_scaling(tensor: torch.Tensor, scale=1e-10) -> torch.Tensor:
    # equivalent to:
    # tf.keras.initializers.VarianceScaling(scale=scale, mode='fan_avg', distribution='uniform')
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    scale /= max(1.0, (fan_in + fan_out) / 2.0)
    limit = math.sqrt(3 * scale)
    return torch.nn.init.uniform_(tensor, -limit, limit)


def initialize_variable_truncated_normal(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to:
    # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    return torch.nn.init.trunc_normal_(tensor, mean=0.0, std=0.02, a=-0.04, b=0.04)


def initialize_variable_glorot_uniform(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to:
    # tf.keras.initializers.GlorotUniform()
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    limit = math.sqrt(6 / (fan_in + fan_out))
    return torch.nn.init.uniform_(tensor, -limit, limit)


def initialize_linear(layer: torch.nn.Linear) -> None:
    initialize_variable_glorot_uniform(layer.weight)
    torch.nn.init.zeros_(layer.bias)
