import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py

filename = 'comap-0010488-2020-01-14-050940.hd5'

sbnames = ['A:LSB', 'A:USB', 'B:LSB', 'B:USB']

feed = 2
sb = 2
freq = 250
n_freq = 4

ncut = 3000

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223, sharex=ax1)
ax4 = fig.add_subplot(224, sharex=ax2)

with h5py.File(filename, mode="r") as my_file:
    tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:-ncut-11500] / 1e6
    tod_sb = my_file['spectrometer/band_average'][feed-1, :, ncut:-ncut-11500] / 1e6
    fr = my_file['spectrometer/frequency'][()]
    t = my_file['spectrometer/MJD'][ncut:-ncut-11500] 
    t = (t - t[0]) * 24 * 60  # minutes
    ts = t * 60  # seconds
for i in range(n_freq):
    # lab = r'feed %i, %s, ch %i' % (feed, sbnames[sb-1], freq + i)
    # lab = r'ch %i' % (freq + i)
    lab = r'$\nu$ = %0.3f GHz' % (fr[sb-1, freq + i])
    ax3.plot(t, tod[sb-1, freq-1+i, :], label=lab, lw=0.2)

# for i in range(4):
lab = r'feed %i, %s' % (feed, sbnames[sb-1])
ax1.plot(t, tod_sb[sb-1, :], 'k', label=lab, lw=0.2)

ax1.text(0.5, 0.1434, 'constant elevation scan', rotation=0, verticalalignment='center', fontsize=11)
ax3.set_xlabel('time [m]')

leg = ax1.legend(loc='lower right')

for line in leg.get_lines():
    line.set_linewidth(1.0)
ax1.set_xlim(t[0], t[-1])

filename = 'comap-0010515-2020-01-15-050844.hd5'

with h5py.File(filename, mode="r") as my_file:
    tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:-ncut] / 1e6
    tod_sb = my_file['spectrometer/band_average'][feed-1, :, ncut:-ncut] / 1e6
    t = my_file['spectrometer/MJD'][ncut:-ncut] 
    t = (t - t[0]) * 24 * 60  # minutes
    ts = t * 60  # seconds
for i in range(n_freq):
    # lab = r'feed %i, %s, ch %i' % (feed, sbnames[sb-1], freq + i)
    lab = r'$\nu$ = %0.2f' % (fr[sb-1, freq + i])
    ax4.plot(t, tod[sb-1, freq-1+i, :], label=lab, lw=0.2)

# for i in range(4):
lab = r'feed %i, %s' % (feed, sbnames[sb-1])
ax2.plot(t, tod_sb[sb-1, :], 'k', label=lab, lw=0.2)
ax2.text(0.5, 0.142, 'lissajous scan', rotation=0, verticalalignment='center', fontsize=11)
ax4.set_xlabel('time [m]')

# ax2.legend(loc='lower right')
ax2.set_xlim(t[0], t[-1])

ax1.set_ylabel(r'power [MW Hz${}^{-1}$]')
ax2.set_ylabel(r'power [MW Hz${}^{-1}$]')
ax3.set_ylabel(r'power [MW Hz${}^{-1}$]')
ax4.set_ylabel(r'power [MW Hz${}^{-1}$]')

use_log = False
if use_log:
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')

leg = ax3.legend(loc='lower right')
for line in leg.get_lines():
    line.set_linewidth(1.0)
# ax4.legend(loc='lower right')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax3.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)
plt.subplots_adjust(wspace=0.23)
plt.savefig('raw_tod.pdf', bbox_inches='tight')
plt.show()
