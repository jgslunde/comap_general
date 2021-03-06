import h5py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

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
            array_time     = np.array(f["/hk/array/frame/utc"])
            tod            = np.array(f["/spectrometer/band_average"])
            tod            = np.nanmean(tod, axis=(1))
            tod_time       = np.array(f["/spectrometer/MJD"])
            feeds = np.array(f["/spectrometer/feeds"])
            vane_active    = array_features&(2**13) != 0
            # vane_time = array_time
        except:
            continue
        if np.sum(vane_active) < 5:
            continue
        stepsize = vane_time[1] - vane_time[0]
        # vane_time -= stepsize
        # print("timestep size = ", vane_time[1] - vane_time[0])
        # print("vane/array time offset = ", (vane_time[0] - array_time[0]))
        reduction_idx = vane_active
        vane_angles = vane_angles[reduction_idx]
        vane_time = vane_time[reduction_idx]

        N = len(vane_time)
        # print(N)

        data[0, i:i+N] = vane_angles

        for feed_idx in range(len(feeds)):
            feed = feeds[feed_idx]

            tod_reduced = np.zeros_like(vane_time)
            for j in range(len(tod_reduced)):
                nearest_idx = np.argmin(np.abs((tod_time+offset) - vane_time[j]))
                tod_reduced[j] = np.nanmean(tod[feed_idx][nearest_idx-10 : nearest_idx+11])
            # tod_spline = interp1d(tod_time+offset, tod[feed_idx], kind="nearest")
            # tod_reduced = tod_spline(vane_time)


            tod_reduced /= np.max(tod_reduced)

            data[feed, i:i+N] = tod_reduced

        i += N

data = data[:, :i]

np.save("angles_0.npy", data)
