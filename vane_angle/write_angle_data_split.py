import h5py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

data1 = np.zeros((21, int(1e7)))
data2 = np.zeros((21, int(1e7)))
months = ["2019-07"]#, "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"] 
paths = ["../../../pathfinder/ovro/" + month + "/" for month in months]
print(paths)

offset = 0

i1 = 0
i2 = 0

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
        except:
            continue
        if np.sum(vane_active) < 5:
            continue

        reduction_idx = vane_active
        vane_angles = vane_angles[reduction_idx]
        vane_time = vane_time[reduction_idx]

        norm1 = 0
        norm2 = 0
        N1 = 0
        N2 = 0

        for feed_idx in range(len(feeds)):
            feed = feeds[feed_idx]

            for j in range(len(vane_time)):
                nearest_idx = np.argmin(np.abs((tod_time+offset) - vane_time[j]))
                if nearest_idx > len(tod_time)//2:
                    data1[0,i1] = vane_angles[j]
                    data1[feed,i1] = np.nanmean(tod[feed_idx][nearest_idx-10 : nearest_idx+11])
                    if data1[feed,i1] >= norm1:
                        norm1 = data1[feed,i1]                    
                    N1 += 1
                else:
                    data2[0,i2] = vane_angles[j]
                    data2[feed,i2] = np.nanmean(tod[feed_idx][nearest_idx-10 : nearest_idx+11])
                    if data2[feed,i2] >= norm2:
                        norm2 = data2[feed,i2]                    
                    N2 += 1
            data1[feed, i1:i1+N1] /= norm1
            data2[feed, i2:i2+N2] /= norm2
            
            i1 += N1
            i2 += N2

            # tod_spline = interp1d(tod_time+offset, tod[feed_idx], kind="nearest")
            # tod_reduced = tod_spline(vane_time)

data1 = data1[:, :i1]
data2 = data2[:, :i2]

np.save("angles1_0.npy", data1)
np.save("angles2_0.npy", data2)
