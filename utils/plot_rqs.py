import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn as nn

i = 1

B = 1.2
eps = 1e-6

N_h_norm = torch.from_numpy(np.loadtxt("h.txt"))
N_w_norm = torch.from_numpy(np.loadtxt("w.txt"))
N_d_pad = torch.from_numpy(np.loadtxt("d.txt"))

N, n_segments = N_h_norm.shape

#fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig, axes = plt.subplots()
N = 1
axes = np.array([axes,])


for n, ax in zip(range(N), axes.flatten()):

    x_b = torch.linspace(-1.1 * B, 1.1 * B, 100, dtype=torch.float64)
    phi_b = torch.empty_like(x_b)

    # Apply mask for linear tails
    inside_mask = abs(x_b) <= B
    x_b_in = x_b[inside_mask]
    phi_b[~inside_mask] = x_b[~inside_mask]

    K = x_b_in.shape[0]
    h_norm = N_h_norm[n]
    w_norm = N_w_norm[n]
    d_pad = N_d_pad[n]

    x_knot_points = (
        torch.cat(
            (torch.tensor([-eps], dtype=torch.float64), torch.cumsum(w_norm, dim=0),), dim=0,
        )
        - B
    )
    phi_knot_points = (
        torch.cat(
            (torch.tensor([-eps], dtype=torch.float64), torch.cumsum(h_norm, dim=0),), dim=0,
        )
        - B
    )

    k_ind = (
        np.searchsorted(x_knot_points.contiguous(), x_b_in.contiguous(),) - 1
    ).clamp(0, n_segments - 1)

    w_k = w_norm[k_ind]
    h_k = h_norm[k_ind]
    s_k = h_k / w_k
    d_k = d_pad[k_ind]
    d_kp1 = d_pad[k_ind + 1]

    x_km1 = x_knot_points[k_ind]
    phi_km1 = phi_knot_points[k_ind]

    alpha = (x_b_in - x_km1) / w_k

    phi_b[inside_mask] = (
        phi_km1
        + (h_k * (s_k * alpha.pow(2) + d_k * alpha * (1 - alpha)))
        / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha))
    )

    ax.plot(x_b, phi_b)
    if n == 0:
        ax.scatter(x_knot_points, phi_knot_points, s=8, c='r', label="knots")
    else:
        ax.scatter(x_knot_points, phi_knot_points, s=8, c='r')

    ax.set_xlabel("$y_{i-1}$")
    ax.set_ylabel("$y_{i}$")

fig.legend()
#fig.suptitle("Example RQS transformations")
fig.savefig("example_rqs.png")
plt.show()
