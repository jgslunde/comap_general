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
vane_active    = array_features&(2**13) != 0
print("timestep size = ", vane_time[1] - vane_time[0])
print("vane/array time offset = ", (vane_time[0] - array_time[0]))

print(tod_time[0]*3600*24, vane_time[0]*3600*24)

vane_angles_reduced = vane_angles[vane_active]
vane_time_reduced   = vane_time[vane_active]

feeds = [8, 1, 12, 16, 14]

fig, ax1 = plt.subplots()
for feed in feeds:
    ax1.plot(tod_time*3600*24, tod[feed-1]/np.max(tod[feed-1]), lw=1, label="FEED=%d" % feed)
ax1.set_ylabel("Normalized TOD")
ax1.set_xlabel("Time")
ax1.legend(loc=4)

ax2 = ax1.twinx()
ax2.set_ylabel("Degrees")
ax2.plot(vane_time_reduced*3600*24, vane_angles_reduced, "ko", label="vane angle")
ax2.legend(loc=3)
ax1.set_title(filename.strip("comp_comap-").strip(".hd5") + " | Band average")
plt.show()