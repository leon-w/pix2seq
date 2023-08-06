import os

import numpy as np

os.environ["KERAS_BACKEND"] = "torch"

import keras_core as keras
import torch

from rin_pytorch.utils.debug_utils import track
from rin_pytorch.utils.lamb_custom import LambCustom

model = keras.layers.Dense(10, use_bias=False)
with torch.no_grad():
    model(torch.zeros(1, 10).cuda())


rng = np.random.default_rng(42)

for w in model.weights:
    w.assign(rng.normal(size=w.shape))

for _ in range(10):
    input_np = rng.normal(size=(8, 10)).astype(np.float32)
    input = torch.from_numpy(input_np).cuda()

    output = model(input)
    track("output", _=output)
    loss = torch.nn.functional.mse_loss(output, input)
    loss.backward()

    print(loss.item())

    for w in model.weights:
        track("grad", g=w.value.grad)

    optimizer = LambCustom(model.parameters(), lr=1e-1, clamp_value=1e6, eps=1e-8, weight_decay=1e-2)
    optimizer.step()
    optimizer.zero_grad()

    for w in model.weights:
        track("w", _=w.value)

# optimizer = LambCustom(model.parameters(), lr=1e-3, weight_decay=1e-2)
# optimizer.step()
