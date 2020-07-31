import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

data = np.load("minimum_angles.npy")
mean1 = 0
mean2 = 0
plt.figure(figsize=(10,8))
for k in range(12):
    angles = []
    i = 0
    while (data[k,i] != -1):
        angles.append(data[k,i])
        i += 1
    angles = np.array(angles)
    angles = angles[angles < 67.5]
    angles = angles[angles > 64.5]

    plt.scatter([k], [np.mean(angles)], s=100, c="r")
    plt.scatter(np.zeros_like(angles)+k, angles, s=20, c="k", alpha=0.05)

plt.title("Minimum vane angle of individual obsids")
plt.legend(["mean of month", "min angles"])
plt.ylim(64.5, 67.5)
plt.ylabel("Vane angle [degrees]")
plt.xticks(range(12), ["2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"], rotation=45)
plt.tight_layout()
plt.savefig("plots/angle_minima.png", bbox_inches="tight")
plt.close()
plt.clf()


from scipy.signal.windows import gaussian

data = np.load("minimum_angles_datetime.npy")
data = data[:,data[0] > 10000]
data[0] = data[0] - np.min(data[0])
data[1] = data[1,np.argsort(data[0])]
data[2] = data[2,np.argsort(data[0])]
data[0] = data[0,np.argsort(data[0])]

clean_idx = data[2] > 4000
data = data[:,clean_idx]

N = len(data[0])
asdf = np.zeros(N)
std = 200
gauss = gaussian(N, std)
data_padded = np.zeros(N*2)
data_padded[N//2:(3*N)//2] = data[1]
data_padded[:N//2] = data_padded[N//2]
data_padded[(3*N)//2:] = np.mean(data_padded[(3*N)//2-30:(3*N)//2])
for i in range(len(asdf)):
    asdf[i] = np.sum(data_padded[i:i+N]*gauss)/(std*np.sqrt(2*np.pi))

fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(data[2], data[1], c="k", s=4, alpha=1, label="min angles")
ax.set_ylim(64.5, 67.5)
ax.set_title("Minimum vane angle of individual obsids")
ax.set_ylabel("Vane angle [degrees]")
# ax.set_xlabel("Days after 2019-07-01")
# ax.set_ylabel("Days after 2019-07-01")
ax.set_xlabel("Obsid")
ax.plot(data[2], asdf, lw=4, c="r", label="Gaussian window")


# secax = ax.secondary_xaxis('top', functions=(forward, inverse))
# secax.set_xlabel('period [s]')
# print(data.shape)
# ax2 = ax.twiny()
# ax2.set_xticks(ticks=data[0,::1000], labels=data[2,::1000])
# ax2.scatter(data[2], data[1], c="k", s=4, alpha=1)
# ax2.set_xlabel("Obsid")
plt.legend()
plt.tight_layout()
# plt.savefig("plots/angle_minima_datetime.png", bbox_inches="tight")
# plt.close()
# plt.clf()
plt.show()