import h5py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

data = np.zeros((19, int(1e7)))
path = "../pathfinder/ovro/2020-04/"

offset = 6e-6

filenames = []
for file in os.listdir(path):
    if file.endswith(".hd5"):
        filenames.append(file)

i = 0
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
        vane_active    = array_features&(2**13) != 0
        # vane_time = array_time
    except:
        continue
    if np.sum(vane_active) < 60:
        continue
    stepsize = vane_time[1] - vane_time[0]
    # vane_time -= stepsize
    print("timestep size = ", vane_time[1] - vane_time[0])
    print("vane/array time offset = ", (vane_time[0] - array_time[0]))

    vane_angles_reduced = vane_angles[vane_active]
    vane_time_reduced   = vane_time[vane_active]
    N = len(vane_time_reduced)
    print(N)

    data[0, i:i+N] = vane_angles_reduced

    for feed in range(1, 19):
        tod_spline = interp1d(tod_time+offset, tod[feed-1], kind="nearest")
        tod_reduced = tod_spline(vane_time_reduced)
        tod_reduced /= np.max(tod_reduced)

        data[feed, i:i+N] = tod_reduced

    i += N

data = data[:, :i]

np.save("angles_6e6.npy", data)