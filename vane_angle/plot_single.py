import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

months = ["2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"] 
paths = ["../../../pathfinder/ovro/" + month + "/" for month in months]
filenames = []
for path in paths:
    for file in os.listdir(path):
        if file.endswith(".hd5"):
            filenames.append(path+file)

np.random.seed(33)

for asdf in tqdm(range(40)):
    filename = np.random.choice(filenames)
    name = filename.split("/")[-1].strip("comp_comap-").strip(".hd5")

    f = h5py.File(filename, "r")
    try:
        tod            = np.array(f["/spectrometer/band_average"])
        tod            = np.nanmean(tod, axis=(1))
        tod_freqs      = np.array(f["/spectrometer/tod"][0,:,:])
        tod_time       = np.array(f["/spectrometer/MJD"])
        vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
        vane_time      = np.array(f["/hk/antenna0/vane/utc"])
        array_features = np.array(f["/hk/array/frame/features"])
        array_time     = np.array(f["/hk/array/frame/utc"])
        all_feeds      = np.array(f["/spectrometer/feeds"])
        vane_active    = array_features&(2**13) != 0
    except:
        continue

    if np.sum(vane_active) > 5:
        vane_angles = vane_angles[vane_active]
        vane_time   = vane_time[vane_active]

        tod_temp = []
        tod_time_temp = []
        for k in range(len(tod_time)):
            if np.min(np.abs(tod_time[k] - vane_time)) < 5e-6:  # ~5 seconds.
                tod_temp.append(tod[:,k])
                tod_time_temp.append(tod_time[k])
        tod_time = np.array(tod_time_temp)
        tod = np.array(tod_temp).T

        feeds = [1, 8, 12, 16, 14]

        fig, ax1 = plt.subplots(figsize=(10,8))
        for feed_idx in feeds:
            feed = all_feeds[feed_idx]
            ax1.plot(tod_time, tod[feed_idx-1]/np.max(tod[feed_idx-1]), lw=2, label="FEED=%d" % feed)
        ax1.set_ylabel("Normalized TOD")
        ax1.set_xlabel("Time [MJD]")
        ax1.legend(loc=4)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Degrees")
        ax2.plot(vane_time, vane_angles, "ko", label="vane angle")
        ax2.legend(loc=3)
        ax1.set_title(name + " | Band average")
        angle_min_time = vane_time[np.argmin(vane_angles)]
        plt.xlim(angle_min_time - 0.00015, angle_min_time + 0.00015)
        plt.tight_layout()
        plt.savefig("plots/tod_angle_%s.png" % name, bbox_inches="tight")
        plt.close()
        plt.clf()



# for feed_idx in feeds:
#     feed = all_feeds[feed_idx]
#     tod_spline = interp1d(tod_time, tod[feed_idx], kind="nearest")
#     tod_reduced = tod_spline(vane_time)
#     tod_reduced /= np.max(tod_reduced)

#     plt.scatter(vane_angles, tod_reduced, label="FEED=%d" % feed)

# plt.legend()
# plt.title(filename.strip("comp_comap-").strip(".hd5"))
# plt.xlabel("Vane angle [degrees]")
# plt.ylabel("Normalized TOD")
# plt.tight_layout()
# plt.savefig("power_angle_single.png", bbox_inches="tight")
# plt.close()
# plt.clf()



# fig, ax1 = plt.subplots()
# for sb in range(4):
#     for freq in range(100, 901, 400):
#         ax1.plot(tod_time, tod_freqs[sb][freq]/np.max(tod_freqs[sb][freq]), lw=1, label="sb=%d, freq_nr=%d" % (sb, freq))
# ax1.set_ylabel("Normalized TOD")
# ax1.set_xlabel("Time [MJD]")
# ax1.legend(loc=4)

# ax2 = ax1.twinx()
# ax2.set_ylabel("Degrees")
# ax2.plot(vane_time, vane_angles, "ko", label="vane angle")
# ax2.legend(loc=3)
# ax1.set_title(filename.strip("comp_comap-").strip(".hd5") + " | Feed 1")
# plt.xlim(58946.68763, 58946.68793)
# plt.tight_layout()
# plt.savefig("tod_freq_angle.png", bbox_inches="tight")
# plt.close()
# plt.clf()