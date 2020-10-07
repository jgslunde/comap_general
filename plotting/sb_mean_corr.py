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
nuse = 50 * 60 * 4 # 5 minutes
samprate = 50

with h5py.File(filename, mode="r") as my_file:
    # tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:-ncut] / 1e5
    tod_sb = my_file['spectrometer/band_average'][:, :, ncut:ncut + nuse]
    pixels = np.array(my_file['spectrometer/feeds'][:]) - 1
n_det = 20
n_sb = 4
n_samp = len(tod_sb[0,0])
tod = np.zeros((n_det, n_sb, n_samp))
tod[pixels] = tod_sb
tod[19] *= np.nan
tod_flat = tod.reshape((n_det * n_sb, n_samp))
corr = np.corrcoef(tod_flat)


fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
# im = ax.imshow(np.log10(np.abs(corr)), vmin=-3, vmax=-1,
#                extent=(0.5, n_det + 0.5, n_det + 0.5, 0.5))
# cbar = fig.colorbar(im, ticks=[-1, -2, -3])
# cbar.ax.set_yticklabels(['0.1', '0.01', '0.001'])
vmax = 1.0
im = ax.imshow(corr, vmin=0, vmax=vmax,  #vmin=-1, vmax=1,#vmin=-0.1, vmax=0.1,
                extent=(0.5, n_det + 0.5, n_det + 0.5, 0.5))
cbar = fig.colorbar(im)
cbar.set_label('Correlation')
new_tick_locations = np.array(range(n_det)) + 1
ax.set_xticks(new_tick_locations)
ax.set_yticks(new_tick_locations)
# ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
xl = np.linspace(0.5, n_det + 0.5, n_det * 1 + 1)
# ax.vlines(xl, ymin=0.5, ymax=n_det + 0.5, linestyle='-', color='k', lw=0.1, alpha=1.0)
# ax.hlines(xl, xmin=0.5, xmax=n_det + 0.5, linestyle='-', color='k', lw=0.1, alpha=1.0)
# ax.hlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), xmin=0.5, xmax=n_det + 0.5, linestyle='--', color='k', lw=0.01, alpha=0.1)
# for i in range(n_det):
#     plt.text(xl[i] + 0.7, xl[i] + 0.2, str(i+1), rotation=0, verticalalignment='center', fontsize=2)
ax.set_xlabel('Feed')
ax.set_ylabel('Feed')
plt.xticks(rotation=90)
plt.savefig('sb_mean_corr.pdf', bbox_inches='tight')
plt.show()
