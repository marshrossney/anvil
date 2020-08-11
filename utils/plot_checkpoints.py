import numpy as np
import matplotlib.pyplot as plt
base_dir = "baseline"
inputs = {
    "sig1": r"$\sigma=1.0, b=4.0$",
    "sig0-5": r"$\sigma=0.5, b=2$",
    "sig0-25": r"$\sigma=0.25, b=1.2$",
    "sig0-125": r"$\sigma=0.125, b=1.2$",
    "sig0-06": r"$\sigma=0.0625, b=1.2$",
    
}

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig, ax2 = plt.subplots()
#ax1.set_xlabel("epoch")
ax2.set_xlabel("epoch")
#ax1.set_ylabel("loss + 10.1")
ax2.set_ylabel("acceptance")

#ax1.set_yscale("log")

for source_dir, label in inputs.items():

    path = base_dir + "/" + source_dir + "/"

    #loss_data = np.loadtxt(path + "loss.txt")
    acceptance_data = np.loadtxt(path + "acceptance.txt")

    #ax1.plot(loss_data[:, 0], loss_data[:, 1] + 10.0, label=label)
    ax2.plot(acceptance_data[:, 0], acceptance_data[:, 1], label=label)


#ax1.legend()
ax2.legend()

plt.show()
