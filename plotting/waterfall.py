import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py

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
nuse = 50 * 60 * 1 # 5 minutes
samprate = 50

fig = plt.figure(figsize=(10, 3))
ax1 = fig.add_subplot()


with h5py.File(filename, mode="r") as my_file:
    tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:-ncut] / 1e6
    # tod_sb = my_file['spectrometer/band_average'][feed-1, :, ncut:ncut] / 1e5
    fr = my_file['spectrometer/frequency'][()] 
    t = my_file['spectrometer/MJD'][ncut:-ncut] 
    t = (t - t[0]) * 24 * 60  # minutes
    ts = t * 60  # seconds
# for i in range(n_freq):
lab = r'feed %i, %s, ch %i' % (feed, sbnames[sb-1], freq)
# lab = r'ch %i' % (freq + i)
# tod[(0, 2)]
# np.linspace(0, 2)
# tod[(0, 2), :, :] = tod[(0, 2), ::-1, :]
for i in range(4):
    ax1.plot(fr[i], tod[i].mean(1), label=sbnames[i])
ax1.legend()
ax1.set_ylim(0, 1)
ax1.set_xlim(26, 34)
ax1.set_xlabel('frequency [GHz]')
ax1.set_ylabel(r'power [MW Hz${}^{-1}$]')
plt.savefig('freq_spectrum.pdf', bbox_inches='tight')
plt.show()
