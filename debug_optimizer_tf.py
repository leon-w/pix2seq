import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import LAMB

from debug_utils import track

model = tf.keras.layers.Dense(10, use_bias=False)
model(tf.zeros((1, 10)))


rng = np.random.default_rng(42)

for w in model.weights:
    w.assign(rng.normal(size=w.shape))


for _ in range(10):
    input_np = rng.normal(size=(8, 10)).astype(np.float32)
    input = tf.convert_to_tensor(input_np)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        output = model(input)
        track("output", _=output)
        loss = tf.reduce_mean(tf.square(output - input))
        grads = tape.gradient(loss, model.trainable_weights)

    print(loss.numpy().item())

    for g in grads:
        track("grad", g=g)

    optimizer = LAMB(learning_rate=1e-1, epsilon=1e-8, weight_decay=1e-2)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    for w in model.weights:
        track("w", _=w.value())
