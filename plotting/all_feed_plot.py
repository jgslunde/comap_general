import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py
from matplotlib.pyplot import cm

def decimate_array(arr, factor=16):
    n = len(arr)
    n_after = n - n % factor
    print(n, n_after)

    arr = arr[:n_after]
    arr = arr.reshape(n_after // factor, factor).mean(1)
    return arr

filename = 'comap-0010488-2020-01-14-050940.hd5'

sbnames = ['A:LSB', 'A:USB', 'B:LSB', 'B:USB']

feed = 4
sb = 2
freq = 250
n_freq = 4

ncut = 3500
nuse = 50 * 60 * 20 # 5 minutes
samprate = 50

with h5py.File(filename, mode="r") as my_file:
    # tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:-ncut] / 1e5
    tod_sb = my_file['spectrometer/band_average'][:, :, ncut:ncut + nuse] / 1e6
    pixels = np.array(my_file['spectrometer/feeds'][:]) - 1
    t = my_file['spectrometer/MJD'][ncut:ncut+nuse] 
    t = (t - t[0]) * 24 * 60  # minutes
n_det = 20
n_sb = 4
n_samp = len(tod_sb[0,0])
tod = np.zeros((n_det, n_sb, n_samp))
tod[pixels] = tod_sb
tod[19] *= np.nan
tod[3] *= np.nan
tod[6] *= np.nan


plt.figure(figsize=(5, 4))
# color=cm.rainbow(np.linspace(0,1,n_det))
color=iter(cm.tab20(np.linspace(0,1,n_det)))
for i in range(n_det-1):
    c=next(color)
    plt.plot(t, tod[i, :].mean(0), c=c, label='feed %02i' % (i+1))
    # for j in range(n_sb):
    #     if j == 0:
    #         plt.plot(t, tod[i, j], c=c, label='feed %02i' % (i+1))
    #     else:
    #         plt.plot(t, tod[i, j], c=c)
plt.xlim(t[0], t[-1])
plt.xlabel('time [m]')
plt.ylabel(r'power [MW Hz${}^{-1}$]')
plt.legend(bbox_to_anchor=(1.01, 1.01), fontsize=8)
plt.savefig('all_feed_plot_good.pdf', bbox_inches='tight')
# plt.show()
