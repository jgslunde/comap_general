import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

data = np.load("angles_7e6.npy")

print(data.shape)

feed = 8

plt.scatter(data[0], data[feed], s=2)
plt.axvline(x=70, c="y", ls="--", label="angle threshold = ca 70 deg.")
plt.legend()
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
plt.title("All of 2020-04 | Feed 8 | Offset = 7e-6 MJD")
plt.tight_layout()
plt.savefig("power_angle_all_feed8_7e-6.png", bbox_inches="tight")
plt.close()
plt.clf()