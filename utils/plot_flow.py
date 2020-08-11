import numpy as np
import matplotlib.pyplot as plt
from sys import argv

sigma = 0.5
n_layers = int(argv[1])

fig, axes = plt.subplots(n_layers + 2, figsize=(8, 12), sharex=True)

axes[0].hist(
    np.loadtxt("model_in.txt").flatten(),
    bins=50,
    density=True,
    histtype="step",
    label="model in",
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

# Standard deviations
sigma = np.loadtxt("model_in.txt").std()
axes[0].text(
    0.1,
    0.9,
    f"$\sigma={sigma:.2g}$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0].transAxes,
)
sigma = np.loadtxt("ensemble_out.txt").std()
axes[-1].text(
    0.1,
    0.9,
    f"$\sigma={sigma:.2g}$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[-1].transAxes,
)

for ax in axes:
    ax.set_yticklabels([])
    ax.set_yticks([])

axes[0].legend(loc=1)
axes[-1].legend(loc=1)

fig.tight_layout()

fig.savefig("flow.png")

# plt.show()
