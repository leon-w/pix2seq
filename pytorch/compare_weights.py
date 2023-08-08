from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

file1 = Path("weights_init_pt.npy")
file2 = Path("../rin_cifar10_fresh1_weights.npy")

file1_np = np.load(file1, allow_pickle=True).item()
weights1_np = list(file1_np.values())
names1 = list(file1_np.keys())

file2_np = np.load(file2, allow_pickle=True).item()
weights2_np = list(file2_np.values())
names2 = list(file2_np.keys())

for i, (w1, w2) in tqdm(enumerate(zip(weights1_np, weights2_np)), total=len(weights1_np)):
    data1 = w1.flatten()
    data2 = w2.flatten()

    assert data1.shape == data2.shape

    if np.all(data1 == data2):
        tqdm.write(f"Skipping {names1[i]}")
        continue

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # compare distribution
    axs[0].hist(data1, bins=100, alpha=0.5, label=file1.stem, color="red")
    axs[0].hist(data2, bins=100, alpha=0.5, label=file2.stem, color="green")
    axs[0].legend()

    axs[1].hist(data1, bins=100, alpha=0.5, label=file1.stem, color="red")
    axs[1].legend()

    axs[2].hist(data2, bins=100, alpha=0.5, label=file2.stem, color="green")
    axs[2].legend()

    fig.suptitle(names1[i])
    fig.savefig(f"figs/hist_{i:03d}.png", bbox_inches="tight")

    # close figure
    plt.close(fig)

    if i >= 1:
        break
