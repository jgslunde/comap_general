import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm, trange
import time
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
import scipy.signal
from numpy.fft import rfft, irfft
from scipy.fftpack import fft, ifft, next_fast_len
import sys
sys.path.insert(1, "/mn/stornext/d16/cmbco/comap/jonas/comap_general/tsys")
from tsysmeasure import TsysMeasure


def Wiener_filter(signal, fknee=0.01, alpha=4.0, samprate=50):
    """ Applies a lowpass (Wiener) filter to a signal, and returns the low-frequency result.
        Uses mirrored padding of signal to deal with edge effects.
        Assumes signal is of shape [freq, time].
    """
    N = signal.shape[-1]
    fastlen = next_fast_len(2*N)
    signal_padded = np.zeros((1024, fastlen))
    signal_padded[:,fastlen//2-N:fastlen//2] = signal
    signal_padded[:,fastlen//2:fastlen//2+N] = signal[:,::-1]
    for i in range(1024):
        signal_padded[i] = np.pad(signal_padded[i,fastlen//2-N:fastlen//2+N], (fastlen-2*N)//2, mode="reflect")

    freq_full = np.fft.fftfreq(fastlen)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    return ifft(fft(signal_padded)*W).real[:,fastlen//2-N:fastlen//2]

def Wiener_filter_safe(signal, fknee=0.01, alpha=4.0, samprate=50):
    """ Applies a lowpass (Wiener) filter to a signal, and returns the low-frequency result.
        Uses mirrored padding of signal to deal with edge effects.
        Assumes signal is of shape [freq, time].
    """
    N = signal.shape[-1]
    signal_padded = np.zeros((1024, 2*N))
    signal_padded[:,:N] = signal
    signal_padded[:,N:] = signal[:,::-1]

    freq_full = np.fft.fftfreq(2*N)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    return ifft(fft(signal_padded)*W).real[:,:N]

def PS_1f(freqs, sigma0, fknee, alpha):
    return sigma0**2*(1 + (freqs/fknee)**alpha)

def gen_1f_data(N, sigma0, fknee, alpha, samprate = 50.0, wn=True):
    frequencies = np.fft.rfftfreq(2*N, d=1.0/samprate)
    if wn:
        Ps = sigma0**2*((1 + fknee/frequencies)**alpha)
    else:
        Ps = sigma0**2*((fknee/frequencies)**alpha)
    Ps[0] = 0
    ft = np.random.normal(0, np.sqrt(Ps), N+1)*np.sqrt(2*N)
    data = np.fft.irfft(ft).real
    return data[:N]

def gen_1f_data_nown(N, fknee, alpha, samprate = 50.0):
    frequencies = np.fft.rfftfreq(2*N, d=1.0/samprate)
    Ps = (fknee/frequencies)**alpha
    Ps[0] = 0
    ft = np.random.normal(0, np.sqrt(Ps), N+1)*np.sqrt(2*N)
    data = np.fft.irfft(ft).real
    return data[:N]


def az_el_func(x, g, d, c):
    return g*x[0] + d*x[1] + c


def az_func(x, d, c):
    return d*x + c

def az_el_template(feed, g, d, c):
    return g/np.sin(self.el[feed]*np.pi/180.0) + d*self.az[feed] + c

def az_template(feed, d, c):
    return d*self.az[feed] + c

def az_el_residual(x, freq, sb, feed):
    g, d, c = x
    res = (az_el_template(feed, g, d, c) - self.tod_norm[feed, sb, freq])
    res[np.isnan(res)] = 0
    return res

def az_residual(x, freq, sb, feed):
    d, c = x
    res = (az_template(feed, d, c) - self.tod_norm[feed, sb, freq])
    res[np.isnan(res)] = 0
    return res


def fit_az_template(x):
    signal, az = x
    if np.isfinite(signal).all():
        #(d, c), _ = scipy.optimize.curve_fit(az_func, az, signal, (1.0, 0.0))
        d, c = scipy.optimize.least_squared(az_residual, args=(0))
    else:
        d, c = np.zeros(1024), np.zeros(1024)
    return d, c

class l2gen:
    def __init__(self, verbose=False, debug=False):
        self.samprate = 50
        self.verbose = verbose
        self.debug = debug
        self.load_level2_info_called = False
        self.provide_scan_info_called = False
        self.load_scan_from_file_called = False
        self.find_tsys_called = False
        self.normalize_gain_called = False
        self.subtract_pointing_templates_called = False
        self.polyfilter_TOD_called = False
        self.pca_filter_poly_TOD_called = False
        self.pca_filter_freq_TOD_called = False
        self.frequency_filter_TOD_called = False
        
        
    def load_level2_info(self, level2_filename):
        if self.verbose:
            print(f"Starting level2 file load from {level2_filename}...")
            t0 = time.time()
        l2f = h5py.File(level2_filename, "r")
        self.l2_feed2ind = l2f["pix2ind"][()]
        self.l2_feeds = l2f["pixels"][()]
        self.l2_freqmask = np.array(l2f["freqmask"][()], dtype=bool)
        self.l2_freqmask_full = np.array(l2f["freqmask_full"][()], dtype=bool)
        self.l2_time = l2f["time"][()]
        self.scan_start_mjd = self.l2_time[0]
        self.scan_stop_mjd = self.l2_time[-1]
        self.load_level2_info_called = True
        try:
            l2f = h5py.File(level2_filename[:-3] + "_4_before_mask.h5", "r")
            self.l2_freqmask_full_premask = np.array(l2f["freqmask_full"][()], dtype=bool)
        except:
            self.l2_freqmask_full_premask = np.array(self.l2_freqmask_full[()], dtype=bool)
            print(f"WARNING: Could not find diag file {level2_filename + '_4_before_mask.h5'}. This run will use the full output freqmask. Consider rerunning l2gen with the l2diag param enabled, to produce this file.")
        if self.verbose:
            print(f"Finished level2 file load in {time.time()-t0:.1f} s.")
        
        
    def provide_scan_info(self, scan_start_mjd, scan_stop_mjd):
        self.scan_start_mjd = scan_start_mjd
        self.scan_stop_mjd = scan_stop_mjd
        self.l2_freqmask_full = None
        self.provide_scan_info_called = True
        
        
    def load_scan_from_file(self, filename):
        if self.verbose:
            print(f"Starting level1 file load from {filename}...")
            t0 = time.time()
        self.filename = filename
        f = h5py.File(filename, "r")
        self.tod_times      = f["/spectrometer/MJD"][()]
        self.scan_start_idx = np.argmin(np.abs(self.scan_start_mjd - self.tod_times))
        self.scan_stop_idx = np.argmin(np.abs(self.scan_stop_mjd - self.tod_times)) + 1
        if (self.scan_stop_idx - self.scan_start_idx)%2 != 0:
            self.scan_stop_idx -= 1
        self.tod_times = self.tod_times[self.scan_start_idx:self.scan_stop_idx]

        self.vane_angles    = f["/hk/antenna0/vane/angle"][()]/100.0  # Degrees
        self.vane_times     = f["/hk/antenna0/vane/utc"][()]
        self.freqs          = f["spectrometer/frequency"][()].reshape(4096)
        self.array_features = f["/hk/array/frame/features"][()]
        self.array_time     = f["/hk/array/frame/utc"][()]
        self.az             = f["/spectrometer/pixel_pointing/pixel_az"][:,self.scan_start_idx:self.scan_stop_idx]
        self.el             = f["/spectrometer/pixel_pointing/pixel_el"][:,self.scan_start_idx:self.scan_stop_idx]
        self.tod_times_seconds = (self.tod_times-self.tod_times[0])*24*60*60
        self.tod            = f["/spectrometer/tod"][:,:,:,self.scan_start_idx:self.scan_stop_idx]
        self.feeds          = f["/spectrometer/feeds"][()]
        if self.tod_times[0] > 58712.03706:
            self.Thot       = f["/hk/antenna0/vane/Tvane"][()]
        else:
            self.Thot       = f["/hk/antenna0/env/ambientLoadTemp"][()]
        self.Nfeeds = self.tod.shape[0]
        self.Nsb = self.tod.shape[1]
        self.Nfreqs = self.tod.shape[2]
        self.Ntod = self.tod.shape[-1]

        scan_center_idx_array = np.argmin(np.abs(self.array_time - self.tod_times[self.Ntod//2]))
        if self.array_features[scan_center_idx_array]&2**4:
            self.scantype = "circ"
        elif self.array_features[scan_center_idx_array]&2**5:
            self.scantype = "ces"
        elif self.array_features[scan_center_idx_array]&2**15:
            self.scantype = "liss"
        else:
            raise ValueError("Unknown scan type.")
        if self.l2_freqmask_full is None:
            self.l2_freqmask_full = np.ones((self.Nfeeds, 4, 1024), dtype=bool)
        self.load_scan_from_file_called = True
        if self.verbose:
            print(f"Scan type: {self.scantype}.")
            print(f"Scan duration: {self.tod_times_seconds[-1] - self.tod_times_seconds[0]} s.")
            print(f"Finished level1 file load in {time.time()-t0:.1f} s.")

    def generate_data(self, Ntod, sigma0_T, fknee_T, alpha_T, fknee_g, alpha_g, sigma0_slope):
        self.gain = np.load("/mn/stornext/d16/cmbco/comap/jonas/gain_exmaple.npy")
        self.tsys = np.load("/mn/stornext/d16/cmbco/comap/jonas/tsys_exmaple.npy")
        sigma0_noise = 1.0/np.sqrt(2e9/1024*0.02)  # Radiometer equation.
        signal_freqs_centered = np.linspace(-1, 1, 1024)
        Nfeed = self.tsys.shape[0]
        self.dT_gen = np.zeros((Nfeed,4,Ntod))
        self.dg_gen = np.zeros((Nfeed,4,Ntod))
        self.slope_gen = np.zeros((Nfeed,4,Ntod))
        self.noise_gen = np.zeros((Nfeed,4,1024,Ntod))
        self.tod = np.zeros((Nfeed,4,1024,Ntod))
        F = np.ones((1024, 1))
        P = np.zeros((1024, 2))
        a = np.zeros((1, Ntod))
        m = np.zeros((2, Ntod))
        for feed in range(19):
            for sb in range(4):
                self.dT_gen[feed,sb] = gen_1f_data(Ntod, sigma0_T, fknee_T, alpha_T, wn=True)
                self.dg_gen[feed,sb] = gen_1f_data(Ntod, sigma0_noise, fknee_g, alpha_g, wn=False)
                self.slope_gen[feed,sb] = np.random.normal(0, sigma0_slope, (Ntod))
                self.noise_gen[feed,sb] = np.random.normal(0, sigma0_noise, (1024, Ntod))
                a[0,:] = self.dg_gen[feed,sb]
                m[0,:] = self.dT_gen[feed,sb]
                m[1,:] = self.slope_gen[feed,sb]
                noise = self.noise_gen[feed,sb]
                P[:,0] = 1/self.tsys[feed,sb]
                P[:,1] = signal_freqs_centered/self.tsys[feed,sb]
                self.tod[feed,sb] = (P.dot(m) + F.dot(a) + noise + 1)*self.tsys[:,None]*self.gain[:,None]

    
    def find_tsys(self):
        if self.verbose:
            print("Starting tsys measurement...")
            t0 = time.time()
        Tsys = TsysMeasure()
        Tsys.load_data_from_file(self.filename)
        Tsys.solve()
        Tsys.Tsys_single()
        self.tsys = Tsys.Tsys
        self.find_tsys_called = True
        if self.verbose:
            print(f"Finished tsys measurement in {time.time()-t0:.1f} s.")


    def normalize_gain(self, safe=False):
        if self.verbose:
            print("Starting normalization...")
            t0 = time.time()
        if safe:
            WF = Wiener_filter_safe
        else:
            WF = Wiener_filter
        with mp.Pool() as p:
            tod_lowpass = p.map(WF, self.tod.reshape(4*self.Nfeeds, 1024, self.Ntod))
        tod_lowpass = np.array(tod_lowpass).reshape(self.Nfeeds, 4, 1024, self.Ntod)
        self.tod_norm = self.tod/tod_lowpass - 1
        del(tod_lowpass)
        self.normalize_gain_called = True
        if self.verbose:
            print(f"Finished normalzation in {time.time()-t0:.1f} s.")
    

    def subtract_pointing_templates(self):
        if self.verbose:
            print("Starting pointing template subtraction...")
            t0 = time.time()
        def az_el_template(feed, g, d, c):
            return g/np.sin(self.el[feed]*np.pi/180.0) + d*self.az[feed] + c

        def az_template(feed, d, c):
            return d*self.az[feed] + c

        def az_el_residual(x, freq, sb, feed):
            g, d, c = x
            res = (az_el_template(feed, g, d, c) - self.tod_norm[feed, sb, freq])
            res[np.isnan(res)] = 0
            return res
        
        def az_residual(x, freq, sb, feed):
            d, c = x
            res = (az_template(feed, d, c) - self.tod_norm[feed, sb, freq])
            res[np.isnan(res)] = 0
            return res
        
        tod_pointing_template = np.zeros_like(self.tod)
        self.tod_pointing_subtracted = np.zeros_like(self.tod)
        self.g_opt, self.d_opt, self.c_opt = np.zeros((self.Nfeeds, 4, 1024)), np.zeros((self.Nfeeds, 4, 1024)), np.zeros((self.Nfeeds, 4, 1024))
        g, d, c = 0, 0, 0
        for feed in range(self.Nfeeds):
            for sb in range(4):
                for freq in range(1024):
                    if self.scantype == "ces":
                        #d, c = scipy.optimize.least_squares(az_residual, (d, c), args=(freq, sb, feed)).x                    
                        if np.isfinite(self.tod_norm[feed, sb, freq]).all():
                            (d, c), _ = scipy.optimize.curve_fit(az_func, self.az[feed], self.tod_norm[feed, sb, freq], (d, c))
                        else:
                            d, c = 0, 0
                    else:
                        #g, d, c = scipy.optimize.least_squares(az_el_residual, (g, d, c), args=(freq, sb, feed)).x
                        if np.isfinite(self.tod_norm[feed, sb, freq]).all():
                            (g, d, c), _ = scipy.optimize.curve_fit(az_el_func, (1.0/np.sin(self.el[feed]*np.pi/180.0), self.az[feed]), self.tod_norm[feed, sb, freq], (g, d, c))
                        else:
                            g, d, c = 0, 0, 0
                    self.g_opt[feed,sb,freq], self.d_opt[feed,sb,freq], self.c_opt[feed,sb,freq] = g, d, c
                    tod_pointing_template[feed,sb,freq] = az_el_template(feed, g, d, c)
        self.tod_pointing_subtracted = self.tod_norm - tod_pointing_template
        del(tod_pointing_template)
        self.subtract_pointing_templates_called = True
        if self.verbose:
            print(f"Finished pointing template subtraction in {time.time()-t0:.1f} s.")
    

    def polyfilter_TOD(self, mask=True):
        if self.verbose:
            print("Starting polynomial filter...")
            t0 = time.time()
        tod_polyfit = np.zeros_like(self.tod)
        self.tod_polyfiltered = np.zeros_like(self.tod)
        self.c0_opt, self.c1_opt = np.zeros((self.Nfeeds, 4, self.Ntod)), np.zeros((self.Nfeeds, 4, self.Ntod))
        sb_freqs = np.linspace(-1, 1, 1024)
        for feed in trange(self.Nfeeds):
            if np.sum(self.l2_freqmask_full[feed]) > 0:
                for sb in range(4):
                    #try:
                    #    ydata = self.tod_pointing_subtracted[feed,sb,:,:].copy()
                    #    ydata[~np.isfinite(ydata)] = 0
                    #    ydata[~self.l2_freqmask_full[feed,sb]] = 0
                    #    c1, c0 = np.polyfit(sb_freqs, ydata, 1, w=self.l2_freqmask_full[feed,sb])
                    #except:
                    #    print(f"Polyfilter failed to converge for feed {feed} sb {sb}.")

                    for idx in range(self.Ntod):
                        try:
                            if mask:
                                c1, c0 = np.polyfit(sb_freqs, self.tod_pointing_subtracted[feed,sb,:,idx], 1, w=self.l2_freqmask_full[feed,sb])
                            else:
                                temp = np.ones((1024))
                                temp[:4] = 0
                                temp[-4:] = 0
                                c1, c0 = np.polyfit(sb_freqs, self.tod_pointing_subtracted[feed,sb,:,idx], 1, w=temp)
                        except:
                            c1, c0 = 0, 0
                            #print(f"Polyfit did not converge for feed {feed} sb {sb} idx {idx}")
                        self.c0_opt[feed,sb,idx] = c0
                        self.c1_opt[feed,sb,idx] = c1
        tod_polyfit = self.c1_opt[:,:,None,:]*sb_freqs[None,None,:,None] + self.c0_opt[:,:,None,:]
        self.tod_polyfiltered = self.tod_pointing_subtracted - tod_polyfit
        del(tod_polyfit)
        self.polyfilter_TOD_called = True
        if self.verbose:
            print(f"Finished polynomial filter in {time.time()-t0:.1f} s.")


    def pca_filter_poly_TOD(self, mask=True):
        if self.verbose:
            print("Starting poly PCA filter...")
            t0 = time.time()
        M = self.tod_polyfiltered.reshape(self.Nfeeds*4*1024, self.Ntod)
        if mask:
            M = M[self.l2_freqmask_full.reshape(self.Nfeeds*4*1024), :]
        M = np.dot(M.T, M)
        eigval, eigvec = np.linalg.eigh(M)
        ak = np.sum(self.tod_polyfiltered[:,:,:,:,None]*eigvec[:,-4:], axis=2)
        self.tod_pca_poly_filtered = self.tod_polyfiltered - np.sum(ak[:,:,None]*eigvec[:,-4:], axis=-1)
        self.pca_filter_poly_TOD_called = True
        if self.verbose:
            print(f"Finished poly PCA filter in {time.time()-t0:.1f} s.")

    def pca_filter_freq_TOD(self, mask=True):
        if self.verbose:
            print("Starting freq PCA filter...")
            t0 = time.time()
        M = self.tod_frequency_filtered.reshape(self.Nfeeds*4*1024, self.Ntod)
        if mask:
            M = M[self.l2_freqmask_full.reshape(self.Nfeeds*4*1024), :]
        M = np.dot(M.T, M)
        eigval, eigvec = np.linalg.eigh(M)
        ak = np.sum(self.tod_frequency_filtered[:,:,:,:,None]*eigvec[:,-4:], axis=2)
        self.tod_pca_freq_filtered = self.tod_frequency_filtered - np.sum(ak[:,:,None]*eigvec[:,-4:], axis=-1)
        self.pca_filter_freq_TOD_called = True
        if self.verbose:
            print(f"Finished freq PCA filter in {time.time()-t0:.1f} s.")

    
    def gain_temp_sep(self, y, P, F, sigma0_g, fknee_g, alpha_g):
        freqs = np.fft.rfftfreq(self.Ntod, d=1/self.samprate)
        Cf = PS_1f(freqs, sigma0_g, fknee_g, alpha_g)
        Cf[0] = 1
        
        sigma0_est = np.std(y[:,1:] - y[:,:-1], axis=1)/np.sqrt(2)
        sigma0_est = np.mean(sigma0_est)
        Z = np.eye(self.Nfreqs, self.Nfreqs) - P.dot(np.linalg.inv(P.T.dot(P))).dot(P.T)
        
        RHS = np.fft.rfft(F.T.dot(Z).dot(y))
        z = F.T.dot(Z).dot(F)
        a_bestfit_f = RHS/(z + sigma0_est**2/Cf)
        a_bestfit = np.fft.irfft(a_bestfit_f)
        m_bestfit = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
        
        return a_bestfit, m_bestfit

    def frequency_filter_TOD(self, mask=True, prior="/mn/stornext/d16/cmbco/comap/jonas/Cf_prior_data.hdf5", prior_fknee_fac=1.0):
        if self.verbose:
            print("Starting frequency filter...")
            t0 = time.time()

        if isinstance(prior, str):
            with h5py.File(prior, "r") as f:
                sigma0_prior = f["sigma0_prior"][self.feeds-1]
                fknee_prior = f["fknee_prior"][self.feeds-1]*prior_fknee_fac
                alpha_prior = f["alpha_prior"][self.feeds-1]
        else:
            sigma0_prior, fknee_prior, alpha_prior = prior
        self.tod_frequency_filtered = np.zeros_like(self.tod)
        sb_freqs = np.linspace(-1, 1, 1024)
        P = np.zeros((1024, 2))
        F = np.zeros((1024, 1))
        if self.debug:
            self.P_all = np.zeros((self.Nfeeds,4,1024,2))
            self.F_all = np.zeros((self.Nfeeds,4,1024,1))
            self.m_all = np.zeros((self.Nfeeds,4,2,self.Ntod))
            self.a_all = np.zeros((self.Nfeeds,4,1,self.Ntod))
        for feed in range(self.Nfeeds):
            for sb in range(4):
                y = self.tod_pointing_subtracted[feed,sb].copy()
                P[:,0] = 1/self.tsys[feed,sb]
                P[:,1] = sb_freqs/self.tsys[feed,sb]
                F[:,0] = 1
                if mask:
                    y[self.l2_freqmask_full[feed,sb] == 0] = 0
                    P[self.l2_freqmask_full[feed,sb] == 0, :] = 0
                    F[self.l2_freqmask_full[feed,sb] == 0, :] = 0
                else:
                    y[:4] = 0
                    y[-4:] = 0
                    P[:4] = 0
                    P[-4:] = 0
                    F[:4] = 0
                    F[-4:] = 0
                try:
                    a, m = self.gain_temp_sep(y, P, F, sigma0_prior[feed], fknee_prior[feed], alpha_prior[feed])
                except:
                    print(f"Frequency filter failed to converge for feed {feed} sb {sb}.")
                    a, m = np.zeros((1,self.Ntod)), np.zeros((2,self.Ntod))
                #self.tod_frequency_filtered[feed,sb] = self.tod[feed,sb] - F.dot(a) - P.dot(m)
                self.tod_frequency_filtered[feed,sb] = self.tod_pointing_subtracted[feed,sb] - F.dot(a) - P.dot(m)
                if self.debug:
                    self.P_all[feed,sb] = P
                    self.F_all[feed,sb] = F
                    self.a_all[feed,sb] = a
                    self.m_all[feed,sb] = m
        self.frequency_filter_TOD_called = True
        if self.verbose:
            print(f"Finished frequency filter in {time.time()-t0:.1f} s.")
    

    def decimate_all_TODs(self, mask=True):
        if self.verbose:
            print("Starting decimation...")
            t0 = time.time()
        if self.load_scan_from_file_called:
            self.tod_dec = self.decimate_signal(self.tod, mask=mask)
        if self.normalize_gain_called:
            self.tod_norm_dec = self.decimate_signal(self.tod_norm, mask=mask)
        if self.subtract_pointing_templates_called:
            self.tod_pointing_subtracted_dec = self.decimate_signal(self.tod_pointing_subtracted, mask=mask)
        if self.polyfilter_TOD_called:
            self.tod_polyfiltered_dec = self.decimate_signal(self.tod_polyfiltered, mask=mask)
        if self.pca_filter_poly_TOD_called:
            self.tod_pca_poly_filtered_dec = self.decimate_signal(self.tod_pca_poly_filtered, mask=mask)
        if self.pca_filter_freq_TOD_called:
            self.tod_pca_freq_filtered_dec = self.decimate_signal(self.tod_pca_freq_filtered, mask=mask)
        if self.frequency_filter_TOD_called:
            self.tod_frequency_filtered_dec = self.decimate_signal(self.tod_frequency_filtered, mask=mask)
        if self.verbose:
            print(f"Finished decimation in {time.time()-t0:.1f} s.")
    
    def decimate_signal(self, signal, mask=True):
        weight = 1.0/np.nanvar(signal, axis=-1)
        if mask:
            weight *= self.l2_freqmask_full
        tod_decimated = np.zeros((signal.shape[0],4,64,signal.shape[-1]))
        for freq in range(64):
            tod_decimated[:,:,freq,:] = np.nansum(signal[:,:,freq*16:(freq+1)*16,:]*weight[:,:,freq*16:(freq+1)*16,None], axis=2)
            tod_decimated[:,:,freq,:] /= np.nansum(weight[:,:,freq*16:(freq+1)*16], axis=2)[:,:,None]
        return tod_decimated