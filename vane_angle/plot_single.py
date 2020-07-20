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
tod_freqs      = np.array(f["/spectrometer/tod"][0,:,:])
tod_time       = np.array(f["/spectrometer/MJD"])
vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
vane_time      = np.array(f["/hk/antenna0/vane/utc"])
array_features = np.array(f["/hk/array/frame/features"])
array_time     = np.array(f["/hk/array/frame/utc"])
vane_active    = array_features&(2**13) != 0
print("timestep size = ", vane_time[1] - vane_time[0])
print("vane/array time offset = ", (vane_time[0] - array_time[0]))

vane_angles_reduced = vane_angles[vane_active]
vane_time_reduced   = vane_time[vane_active]

feeds = [8, 1, 12, 16, 14]

for feed in feeds:
    tod_spline = interp1d(tod_time, tod[feed], kind="nearest")
    tod_reduced = tod_spline(vane_time_reduced)
    tod_reduced /= np.max(tod_reduced)

    plt.scatter(vane_angles_reduced, tod_reduced, label="FEED=%d" % feed)

plt.legend()
plt.title(filename.strip("comp_comap-").strip(".hd5"))
plt.xlabel("Vane angle [degrees]")
plt.ylabel("Normalized TOD")
plt.tight_layout()
plt.savefig("power_angle_single.png", bbox_inches="tight")
plt.close()
plt.clf()



fig, ax1 = plt.subplots()
for feed in feeds:
    ax1.plot(tod_time, tod[feed-1]/np.max(tod[feed-1]), lw=1, label="FEED=%d" % feed)
ax1.set_ylabel("Normalized TOD")
ax1.set_xlabel("Time [MJD]")
ax1.legend(loc=4)

ax2 = ax1.twinx()
ax2.set_ylabel("Degrees")
ax2.plot(vane_time_reduced, vane_angles_reduced, "ko", label="vane angle")
ax2.legend(loc=3)
ax1.set_title(filename.strip("comp_comap-").strip(".hd5") + " | Band average")
plt.xlim(58946.68763, 58946.68793)
plt.tight_layout()
plt.savefig("tod_angle.png", bbox_inches="tight")
plt.close()
plt.clf()



fig, ax1 = plt.subplots()
for sb in range(4):
    for freq in range(100, 901, 400):
        ax1.plot(tod_time, tod_freqs[sb][freq]/np.max(tod_freqs[sb][freq]), lw=1, label="sb=%d, freq_nr=%d" % (sb, freq))
ax1.set_ylabel("Normalized TOD")
ax1.set_xlabel("Time [MJD]")
ax1.legend(loc=4)

ax2 = ax1.twinx()
ax2.set_ylabel("Degrees")
ax2.plot(vane_time_reduced, vane_angles_reduced, "ko", label="vane angle")
ax2.legend(loc=3)
ax1.set_title(filename.strip("comp_comap-").strip(".hd5") + " | Feed 1")
plt.xlim(58946.68763, 58946.68793)
plt.tight_layout()
plt.savefig("tod_freq_angle.png", bbox_inches="tight")
plt.close()
plt.clf()