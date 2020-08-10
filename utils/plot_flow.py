import numpy as np
import matplotlib.pyplot as plt

n_layers = 6

fig, axes = plt.subplots(n_layers + 2, figsize=(8, 12), sharex=True)

axes[0].hist(
    np.loadtxt("model_in.txt").flatten(), bins=50, density=True, histtype="step", label="model in"
)
for i in range(1, n_layers + 1):
    axes[i].hist(
        np.loadtxt(f"layer_{i}.txt").flatten(),
        bins=50,
        density=True,
        histtype="step",
        label=f"layer {i}",
    )

axes[-1].hist(
    np.loadtxt("model_out.txt").flatten(),
    bins=50,
    density=True,
    histtype="step",
    label="model out",
)
axes[-1].hist(
    np.loadtxt("ensemble_out.txt").flatten(),
    bins=50,
    density=True,
    histtype="step",
    label="ensemble out",
)

for ax in axes:
    ax.set_yticklabels([])
    ax.set_yticks([])

axes[0].legend()
axes[-1].legend()

fig.tight_layout()

plt.show()
