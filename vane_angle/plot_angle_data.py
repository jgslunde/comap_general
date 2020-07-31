import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

data = np.load("angles_0.npy")

print(data.shape)

feed = 1

plt.scatter(data[0], data[feed], s=0.01)
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
# plt.xlim(60, 220)
plt.title("2019-07 - 2020-06 | Feed 1 | Offset = 0 MJD")
plt.tight_layout()
plt.savefig("power_angle_all_0.png", bbox_inches="tight")
plt.close()
plt.clf()