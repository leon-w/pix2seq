import math

import torch


def initialize_variable(tensor, scale=1e-10):
    # equivalent to:
    # tf.keras.initializers.VarianceScaling(scale=scale, mode='fan_avg', distribution='uniform')
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    scale /= max(1.0, (fan_in + fan_out) / 2.0)
    limit = math.sqrt(3 * scale)
    return torch.nn.init.uniform_(tensor, -limit, limit)
