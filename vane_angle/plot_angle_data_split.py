import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

data1 = np.load("angles11_0.npy")
data2 = np.load("angles12_0.npy")
data3 = np.load("angles21_0.npy")
data4 = np.load("angles22_0.npy")

print(data1.shape, data2.shape, data3.shape, data4.shape)

feed = 8
print(np.sum(np.isnan(data1[feed])), np.sum(np.isnan(data2[feed])), np.sum(np.isnan(data3[feed])), np.sum(np.isnan(data4[feed])))

plt.scatter(data1[0], data1[feed], s=0.2)
# plt.hist(data1[0])
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
# plt.xlim(60, 220)
plt.title("2019-07 - 2020-06 | Feed 1 | Offset = 0 MJD")
plt.tight_layout()
plt.savefig("power_angle_all11_0.png", bbox_inches="tight")
plt.close()
plt.clf()

plt.scatter(data2[0], data2[feed], s=0.2)
# plt.hist(data2[0])
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
# plt.xlim(60, 220)
plt.title("2019-07 - 2020-06 | Feed 1 | Offset = 0 MJD")
plt.tight_layout()
plt.savefig("power_angle_all12_0.png", bbox_inches="tight")
plt.close()
plt.clf()

plt.scatter(data3[0], data3[feed], s=5, alpha=0.1)
# plt.hist(data3[feed])
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
# plt.xlim(60, 220)
plt.title("2019-07 - 2020-06 | Feed 1 | Offset = 0 MJD")
plt.tight_layout()
plt.savefig("power_angle_all21_0.png", bbox_inches="tight")
plt.close()
plt.clf()

plt.scatter(data4[0], data4[feed], s=0.2)
# plt.hist(data4[0])
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
# plt.xlim(60, 220)
plt.title("2019-07 - 2020-06 | Feed 1 | Offset = 0 MJD")
plt.tight_layout()
plt.savefig("power_angle_all22_0.png", bbox_inches="tight")
plt.close()
plt.clf()
