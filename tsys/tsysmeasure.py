# Run as
# export OMP_NUM_THREADS=18; python3 tsysmeasure.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import ctypes

class TsysMeasure:
    verbose = True

    def __init__(self):
        pass

    def load_data_from_arrays(self, vane_angles, vane_times, array_features, T_hot, tod, tod_times):
        if self.verbose:
            print("Starting arrayload")
            t0 = time.time()
        self.vane_angles = vane_angles
        self.vane_times = vane_times
        self.array_features = array_features
        self.Thot_cont = T_hot/100.0 + 273.15
        self.tod_times = tod_times
        self.nr_vane_times = len(vane_times)

        vane_active = array_features&(2**13) != 0
        self.vane_time1 = vane_times[:self.nr_vane_times//2]
        self.vane_time2 = vane_times[self.nr_vane_times//2:]
        self.vane_active1 = vane_active[:self.nr_vane_times//2]
        self.vane_active2 = vane_active[self.nr_vane_times//2:]

        self.nfeeds, self.nbands, self.nfreqs, self.ntod = tod.shape

        self.tod = tod.astype(dtype=np.float32, copy=False)

        self.Thot = np.zeros((self.nfeeds, 2), dtype=np.float64)
        self.Phot = np.zeros((self.nfeeds, self.nbands, self.nfreqs, 2), dtype=np.float64)  # P_hot measurements from beginning and end of obsid.
        self.Phot_unc = np.zeros((self.nfeeds, self.nbands, self.nfreqs, 2), dtype=np.float64)
        self.Phot_t = np.zeros((self.nfeeds, 2), dtype=np.float64)
        self.Phot[:] = np.nan  # All failed calcuations of Tsys should result in a nan, not a zero.
        self.Phot_unc[:] = np.nan
        self.Phot_t[:] = np.nan

        self.points_used_Phot = np.zeros((self.nfeeds, 2))
        self.points_used_Thot = np.zeros((self.nfeeds, 2))
        self.calib_indices_tod = np.zeros((2, 2), dtype=np.int)  # Start and end indices, in tod_time format, for "calibration phase".
        self.successful = np.zeros((self.nfeeds, 2), dtype=bool)

        self.TCMB = 2.725
        if self.verbose:
            print("Finished arrayload in %.2f s" % (time.time()-t0))


    def load_data_from_file(self, filename):
        if self.verbose:
            print("Starting fileread")
            t0 = time.time()
        f = h5py.File(filename, "r")
        vane_angles    = f["/hk/antenna0/vane/angle"][()]/100.0  # Degrees
        vane_times     = f["/hk/antenna0/vane/utc"][()]
        array_features = f["/hk/array/frame/features"][()]
        tod            = f["/spectrometer/tod"][()].astype(dtype=np.float32, copy=False)
        tod_times      = f["/spectrometer/MJD"][()]
        feeds          = f["/spectrometer/feeds"][()]
        if tod_times[0] > 58712.03706:
            T_hot      = f["/hk/antenna0/vane/Tvane"][()]
        else:
            T_hot      = f["/hk/antenna0/env/ambientLoadTemp"][()]
        if self.verbose:
            print("Finished fileread in %.2f s" % (time.time()-t0))
        self.load_data_from_arrays(vane_angles, vane_times, array_features, T_hot, tod, tod_times)


    def solve(self):
        if self.verbose:
            print("Starting Phot solve")
            t0 = time.time()        
        ### Step 1: Calculate P_hot at the start and end Tsys measurement points. ###
        vane_time1, vane_time2, vane_active1, vane_active2, tod, tod_times = self.vane_time1, self.vane_time2, self.vane_active1, self.vane_active2, self.tod, self.tod_times
        nfeeds, nbands, nfreqs, ntod = self.nfeeds, self.nbands, self.nfreqs, self.ntod
        for i, vane_timei, vane_activei in [[0, vane_time1, vane_active1], [1, vane_time2, vane_active2]]:
            if np.sum(vane_activei) > 5:  # If Tsys
                vane_timei = vane_timei[vane_activei]
                tod_start_idx = np.argmin(np.abs(vane_timei[0]-tod_times))
                tod_stop_idx = np.argmin(np.abs(vane_timei[-1]-tod_times))
                self.calib_indices_tod[i, :] = tod_start_idx, tod_stop_idx
                for feed_idx in range(nfeeds):
                    todi = tod[feed_idx, :, :, tod_start_idx:tod_stop_idx]
                    tod_timesi = tod_times[tod_start_idx:tod_stop_idx]
                    tod_freq_mean = np.nanmean(todi, axis=(0,1))
                    if np.sum(tod_freq_mean > 0) > 10:  # Check number of valid points. Also catches NaNs.
                        threshold_idxs = np.argwhere(tod_freq_mean > 0.95*np.max(tod_freq_mean))  # Points where tod is at least 95% of max. (We assume this is only true during Tsys measurement).
                        min_idxi = threshold_idxs[0][0] + 40  # Take the first and last of points fulfilling the above condition, assume they represent start and end of 
                        max_idxi = threshold_idxs[-1][0] - 40  # Tsys measurement, and add a 40-idx safety margin (I think this is 40*20ms = approx 1 second.)
                        min_idx_vane = np.argmin(np.abs(self.vane_times - tod_timesi[min_idxi]))
                        max_idx_vane = np.argmin(np.abs(self.vane_times - tod_timesi[max_idxi]))
                        if max_idxi > min_idxi and max_idx_vane > min_idx_vane:
                            self.Thot[feed_idx, i] = np.nanmean(self.Thot_cont[min_idx_vane:max_idx_vane])
                            self.Phot[feed_idx, :, :, i] = np.nanmean(todi[:,:,min_idxi:max_idxi], axis=(2))
                            self.Phot_unc[feed_idx, :, :, i] = np.nanstd(todi[:,:,min_idxi:max_idxi], axis=(2))/np.sqrt(max_idxi-min_idxi)
                            self.Phot_t[feed_idx, i] = (tod_timesi[min_idxi] + tod_timesi[max_idxi])/2.0
                            self.points_used_Phot[feed_idx, i] = max_idxi - min_idxi
                            self.points_used_Thot[feed_idx, i] = max_idx_vane - min_idx_vane
                            self.successful[feed_idx, i] = True
        if self.verbose:
            print("Finished Phot solve in %.2f s" % (time.time()-t0))

    def Tsys_of_t(self, t, tod):
        if self.verbose:
            print("Starting Tsys solve")
            t0 = time.time()      
        self.Tsys = np.zeros((self.nfeeds, self.nbands, self.nfreqs, self.ntod), dtype=np.float32)
        tsyslib = ctypes.cdll.LoadLibrary("/mn/stornext/d16/cmbco/comap/jonas/comap_general/tsys/tsyslib.so.1")
        float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")
        float64_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags="contiguous")
        float64_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=4, flags="contiguous")

        tsyslib.tsys_calc.argtypes = [float32_array4, float32_array4, float64_array2,
                                    float64_array1, float64_array4, float64_array2,
                                    ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        tsyslib.tsys_calc(self.Tsys, self.tod, self.Thot, t, self.Phot, self.Phot_t, self.TCMB,
                        self.nfeeds, self.nbands, self.nfreqs, self.ntod)
        self.Tsys[:, :, :, self.calib_indices_tod[0,0]:self.calib_indices_tod[0,1]] = np.nan
        self.Tsys[:, :, :, self.calib_indices_tod[1,0]:self.calib_indices_tod[1,1]] = np.nan
        if self.verbose:
            print("Finished Tsys solve in %.2f s" % (time.time()-t0))
        return self.Tsys


if __name__ == "__main__":
    obsid = "0014454"

    tod_in_path = "/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-06/"
    tod_in_filename = tod_in_path + "comap-0014454-2020-06-23-221036.hd5"

    Tsys = TsysMeasure()
    Tsys.load_data_from_file(tod_in_filename)
    Tsys.solve()

    tsys = Tsys.Tsys_of_t(Tsys.tod_times, Tsys.tod)
    np.save("tsys_"+obsid+".npy", Tsys.Tsys)
    np.save("thot_"+obsid+".npy", Tsys.Thot)
    np.save("thot_cont_"+obsid+".npy", Tsys.Thot_cont)
    np.save("phot_t_"+obsid+".npy", Tsys.Phot_t)
    np.save("phot_"+obsid+".npy", Tsys.Phot)
    np.save("pcold_"+obsid+".npy", Tsys.tod)
    np.save("points_used_Thot"+obsid+".npy", Tsys.points_used_Thot)
    np.save("points_used_Phot"+obsid+".npy", Tsys.points_used_Phot)