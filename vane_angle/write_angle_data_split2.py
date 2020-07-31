import h5py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

for division in ["first", "last"]:


    data = np.zeros((21, int(1e7)))
    months = ["2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"] 
    paths = ["../../../pathfinder/ovro/" + month + "/" for month in months]
    print(paths)

    offset = 0

    i = 0

    for path in paths:
        filenames = []
        for file in os.listdir(path):
            if file.endswith(".hd5"):
                filenames.append(file)

        for filename in tqdm(filenames):
            try:
                f = h5py.File(path + filename, "r")
                vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
                vane_time      = np.array(f["/hk/antenna0/vane/utc"])
                array_features = np.array(f["/hk/array/frame/features"])
                tod            = np.array(f["/spectrometer/band_average"])
                tod            = np.nanmean(tod, axis=(1))
                tod_time       = np.array(f["/spectrometer/MJD"])
                feeds = np.array(f["/spectrometer/feeds"])
                vane_active    = array_features&(2**13) != 0
                # vane_time = array_time
            except:
                continue

            half_idx_vane  = len(vane_time)//2
            half_idx_tod   = len(tod_time)//2

            if division == "first":
                vane_angles = vane_angles[:half_idx_vane]
                vane_time   = vane_time[:half_idx_vane]
                vane_active = vane_active[:half_idx_vane]
                tod         = tod[:,:half_idx_tod]
                tod_time    = tod_time[:half_idx_tod]
            else:
                vane_angles = vane_angles[half_idx_vane:]
                vane_time   = vane_time[half_idx_vane:]
                vane_active = vane_active[half_idx_vane:]
                tod         = tod[:,half_idx_tod:]
                tod_time    = tod_time[half_idx_tod:]


            if np.sum(vane_active) < 5:
                continue

            reduction_idx = vane_active
            vane_angles = vane_angles[reduction_idx]
            vane_time = vane_time[reduction_idx]

            N = len(vane_time)

            data[0, i:i+N] = vane_angles

            for feed_idx in range(len(feeds)):
                feed = feeds[feed_idx]

                tod_reduced = np.zeros_like(vane_time)
                for j in range(len(tod_reduced)):
                    nearest_idx = np.argmin(np.abs((tod_time+offset) - vane_time[j]))
                    tod_reduced[j] = np.nanmean(tod[feed_idx][nearest_idx-10 : nearest_idx+11])

                tod_reduced /= np.max(tod_reduced)

                data[feed, i:i+N] = tod_reduced

            i += N

    data = data[:, :i]

    if division == "first":
        np.save("angles1_0.npy", data)
    else:
        np.save("angles2_0.npy", data)