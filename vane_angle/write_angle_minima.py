import h5py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

months = ["2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"] 
paths = ["../../../pathfinder/ovro/" + month + "/" for month in months]
print(paths)

offset = 0
# data = np.zeros((len(paths), int(1e6))) - 1

# j = 0
# for path in paths:
#     i = 0
#     filenames = []
#     for file in os.listdir(path):
#         if file.endswith(".hd5"):
#             filenames.append(file)

#     for filename in tqdm(filenames):
#         try:
#             f = h5py.File(path + filename, "r")
#             vane_angles    = np.array(f["/hk/antenna0/vane/angle"])/100.0  # Degrees
#             array_features = np.array(f["/hk/array/frame/features"])
#             vane_active    = array_features&(2**13) != 0
#         except:
#             continue
#         if np.sum(vane_active) < 5:
#             continue

#         reduction_idx = vane_active
#         vane_angles = vane_angles[reduction_idx]
#         data[j,i] = np.min(vane_angles)
#         i += 1
#     print(np.mean(data[j,:i]))
#     j += 1

# np.save("minimum_angles.npy", data)


data = np.zeros((3, int(1e6)))

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
            vane_active    = array_features&(2**13) != 0
        except:
            continue
        if np.sum(vane_active) < 5:
            continue
        try:
            reduction_idx = vane_active
            vane_angles = vane_angles[reduction_idx]
            vane_time = vane_time[reduction_idx]
            data[0,i] = np.min(vane_time)
            data[1,i] = np.min(vane_angles)
            data[2,i] = filename.split("-")[-5]
            i += 1
        except:
            pass

data = data[:,:i]
np.save("minimum_angles_datetime.npy", data)
