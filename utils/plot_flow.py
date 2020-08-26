import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit


def load(fname, pos=False):
    data = np.loadtxt(fname)
    if pos:
        condition = data.sum(axis=1) > 0
        return data[condition].flatten()
    else:
        return data.flatten()


if "last" in argv:
    fig, ax = plt.subplots()
    ax.hist(
        load("model_out.txt", pos=False),
        bins=50,
        density=True,
        histtype="step",
        label="Full output",
    )
    ax.hist(
        load("model_out.txt", pos=True),
        bins=50,
        density=True,
        histtype="step",
        label=r"$\langle \phi \rangle > 0$",
    )
    
    sigma = load("model_out.txt", pos=False).std()
    sigma_plus = load("model_out.txt", pos=True).std()
    ax.text(
    0.1,
    0.9,
    f"$\sigma={sigma:.2g}$\n$\sigma_\pm={sigma_plus:.2g}$",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax.transAxes,
    )
    
    ax.legend(loc=5)
    fig.tight_layout()
    fig.savefig("model.png")

    exit()

n_layers = int(argv[1])

fig, axes = plt.subplots(n_layers + 2, figsize=(8, 12), sharex=True)

axes[0].hist(
    load("model_in.txt"), bins=50, density=True, histtype="step", label="model in",
)
for i in range(1, n_layers + 1):
    axes[i].hist(
        load(f"layer_{i}.txt"),
        bins=50,
        density=True,
        histtype="step",
        label=f"layer {i}",
    )

axes[-1].hist(
    load("model_out.txt"), bins=50, density=True, histtype="step", label="model out",
)
axes[-1].hist(
    load("ensemble_out.txt"),
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
