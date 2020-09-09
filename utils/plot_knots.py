import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn as nn

n_layers = 1

fig, axes = plt.subplots(ncols=n_layers, sharex=True, sharey=True)
axes = [axes,]

for i in range(1, n_layers + 1):
    ax = axes[i-1]

    x_kp = np.loadtxt(f"x_kp_{i+1}.txt")
    phi_kp = np.loadtxt(f"phi_kp_{i+1}.txt")

    ax.errorbar(
        x_kp.mean(axis=0),
            phi_kp.mean(axis=0),
            xerr=x_kp.std(axis=0),
            yerr=phi_kp.std(axis=0),
            marker="o",
            markersize=4,
            linestyle="",
        )

    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])

    #ax.set_title(f"layer {i}")
    ax.set_xlabel(f"$y_{i-1}$")
    ax.set_ylabel(f"$y_{i}$")

    ax.set(adjustable="box", aspect="equal")  # want square box
    
fig.tight_layout()

fig.savefig("knots.png")

#plt.show()
