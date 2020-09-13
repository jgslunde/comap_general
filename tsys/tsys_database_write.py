import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib
import time
from tqdm import trange
from tsysmeasure import TsysMeasure
from multiprocessing import Pool

path = "/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-06/"
filenames = []
for f in listdir(path):
   if isfile(join(path, f)):
       if f[-4:] == ".hd5":
           filenames.append(f)


def func(fileidx):
    t0 = time.time()
    filename = filenames[fileidx]
    obsid = filename.split("-")[1]
    print("Starting file nr %d, %s" % (fileidx, filename))
    try:
        f = h5py.File(path + filename, "r")
        freqs = f["spectrometer/frequency"][()]
        vane_angles    = f["/hk/antenna0/vane/angle"][()]/100.0  # Degrees
        vane_times     = f["/hk/antenna0/vane/utc"][()]
        array_features = f["/hk/array/frame/features"][()]
        tod            = f["/spectrometer/tod"][()]
        tod_times      = f["/spectrometer/MJD"][()]
        feeds          = f["/spectrometer/feeds"][()]
        if tod_times[0] > 58712.03706:
            T_hot      = f["/hk/antenna0/vane/Tvane"][()]
        else:
            T_hot      = f["/hk/antenna0/env/ambientLoadTemp"][()]

        print("Loading data from file %d" % fileidx)
        Tsys = TsysMeasure()
        Tsys.load_data_from_arrays(vane_angles, vane_times, array_features, T_hot, tod, tod_times)
        print("Solving Phot %d" % fileidx)
        Tsys.solve()
        print("Solving Tsys %d" % fileidx)
        Tsys.Tsys_of_t(Tsys.tod_times, Tsys.tod)
        print("Calc Tsysmean %d" % fileidx)
        del(tod)  # Deleting tod from memory now that it's not needed, reducing memory load by 1xfilesize.
        del(Tsys.tod)  # Deleting tod from memory now that it's not needed, reducing memory load by 1xfilesize.
        _tsys_timeavg = np.nanmean(Tsys.Tsys, axis=(3))  # 1xfilesize additional memory load here.

        print("Setting up arrays %d" % fileidx)
        Thot = np.zeros((19,2))
        Phot = np.zeros((19,4,1024,2))
        points_used_Thot = np.zeros((19, 2))
        points_used_Phot = np.zeros((19, 2))
        calib_times = np.zeros((19,2))
        tsys_timeavg = np.zeros((19,4,1024))
        for i in range(len(feeds)):
            feed = feeds[i]
            if feed < 20:
                Thot[feed-1] = Tsys.Thot[i]
                Phot[feed-1] = Tsys.Phot[i]
                points_used_Thot[feed-1] = Tsys.points_used_Thot[i]
                points_used_Phot[feed-1] = Tsys.points_used_Phot[i]
                calib_times[feed-1] = Tsys.Phot_t[i]
                tsys_timeavg[feed-1] = _tsys_timeavg[i]
        
        Thot = Thot.reshape((2,19))
        Phot = Phot.reshape((2,19,4,1024))
        points_used_Thot = points_used_Thot.reshape((2,19))
        points_used_Phot = points_used_Phot.reshape((2,19))
        calib_times = calib_times.reshape((2,19))

        print("Finished nr %d in %.2f m" % (fileidx, (time.time()-t0)/60.0))

    except:
        print("!!!!!! Failed %d" % fileidx)
        Thot = np.full((2,19), np.nan)
        Phot = np.full((2,19,4,1024), np.nan)
        points_used_Thot = np.full((2, 19), np.nan)
        points_used_Phot = np.full((2, 19), np.nan)
        calib_times = np.full((2, 19), np.nan)
        tsys_timeavg = np.full((19,4,1024), np.nan)
        freqs = np.full((1024,4), np.nan)
        
    return Thot, Phot, points_used_Thot, points_used_Phot, calib_times, obsid, freqs, tsys_timeavg



with Pool(processes=12) as pool:
    results = pool.map(func, range(len(filenames)))


fout = h5py.File("test.h5", "w")
for result in results:
    Thot, Phot, points_used_Thot, points_used_Phot, calib_times, obsid, freqs, tsys_timeavg = result
    datagroup = "obsid/" + obsid + "/"
    fout.create_dataset(datagroup + "Thot", data=Thot)
    fout.create_dataset(datagroup + "Phot", data=Phot)
    fout.create_dataset(datagroup + "points_used_Thot", data=points_used_Thot)
    fout.create_dataset(datagroup + "points_used_Phot", data=points_used_Phot)
    fout.create_dataset(datagroup + "Tsys_timeavg", data=tsys_timeavg)
    fout.create_dataset(datagroup + "frequencies", data=freqs)
fout.close()