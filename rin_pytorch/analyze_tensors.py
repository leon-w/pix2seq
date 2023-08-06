from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from colorama import Fore, Style


def red(s):
    return f"{Fore.LIGHTRED_EX}{s}{Style.RESET_ALL}"


def green(s):
    return f"{Fore.GREEN}{s}{Style.RESET_ALL}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=Path, default="tensors")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    files_tf = sorted(args.path.glob("TF-*.npy"))
    files_pt = sorted(args.path.glob("PT-*.npy"))

    for tensor_tf, tensor_pt in zip(files_tf, files_pt):
        assert tensor_tf.name[3:] == tensor_pt.name[3:], f"{tensor_tf.name} != {tensor_pt.name}"

        name = tensor_tf.name[3:-4]

        index = int(name.split("-")[1])
        if index < args.skip:
            continue

        tf = np.load(tensor_tf)
        pt = np.load(tensor_pt)

        if len(tf.shape) == 4:
            # adjust channel order for images
            tf = tf.transpose((0, 3, 1, 2))

        if tf.shape == pt.shape:
            diff = np.mean(np.abs(tf - pt))
            c = green if diff < 1e-5 else red
            print(c(f"{name:<15}: {diff:.14f}"))
        else:
            print(f"{name}: {tf.shape} != {pt.shape}")
