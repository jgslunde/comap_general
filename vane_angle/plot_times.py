import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

path = "../pathfinder/ovro/2020-04/"
filename1 = "comp_comap-0012507-2020-04-07-162941.hd5"
filename2 = "comp_comap-0012556-2020-04-09-102442.hd5"

filename = filename1

f = h5py.File(path + filename, "r")

tod            = np.array(f["/spectrometer/band_average"])
tod            = np.nanmean(tod, axis=(1))
tod_time       = np.array(f["/spectrometer/MJD"])
vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
vane_time      = np.array(f["/hk/antenna0/vane/utc"])
array_features = np.array(f["/hk/array/frame/features"])
array_time     = np.array(f["/hk/array/frame/utc"])

vane_time_norm = (vane_time - tod_time[0])*(3600*24)
array_time_norm = (array_time - tod_time[0])*(3600*24)
tod_time_norm = (tod_time - tod_time[0])*(3600*24)

offset = 7e-6
offset_norm = offset*(3600*24)

idx = 2
plt.figure(figsize=(10,3))
plt.scatter(tod_time_norm[:idx*25], np.zeros(idx*25)+0.6, label="spectrometer/MJD")
plt.scatter(vane_time_norm[:idx], np.zeros(idx), label="antenna0/vane/utc")
plt.scatter(array_time_norm[:idx], np.zeros(idx)-0.6, label="array/frame/utc")
plt.plot([tod_time_norm[0], tod_time_norm[0]+offset_norm], [1.2, 1.2], lw=3, c="k", label="offset")
plt.legend(loc=(0.8, 0.65))
plt.yticks([])
plt.xlabel("Time [seconds]")
plt.ylim(-0.8, 2.0)
plt.tight_layout()
plt.title(filename.strip("comp_comap-").strip(".hd5") + " | Offset of 7e-6 MJD (0.6 sec)")
plt.savefig("times_2.png", bbox_inches="tight")
plt.close()
plt.clf()