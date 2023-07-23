import numpy as np


def test_torch():
    from modules.DropPath import DropPath

    import torch

    x = torch.rand(8, 512, 256)

    droptPath = DropPath(0.1)

    y = droptPath(x)
    print(y.shape)


def test_tf():
    import tensorflow as tf
    from modules_tf.DropPath_tf import DropPath_tf

    x = tf.random.uniform((8, 512, 256))

    droptPath = DropPath_tf(0.1)

    y = droptPath(x, training=True)
    # print(y)


test_torch()
