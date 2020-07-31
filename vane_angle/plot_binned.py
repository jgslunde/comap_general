import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

# feed = 9
data = np.load("angles_7e-6.npy")

print(data.shape)

for k, feed in enumerate([8,10]):

    clean_idx = ~np.isnan(data[feed])
    signal = data[feed][clean_idx]
    deg = data[0][clean_idx]
    signal = signal[np.argsort(deg)]
    deg = deg[np.argsort(deg)]
    signal = signal[deg > 65]
    deg = deg[deg > 65]

    points_per_deg = 4
    deg_bins = np.linspace(65, 220, 155*points_per_deg + 1)
    deg_bin_centers = np.linspace(65 + 1.0/(2.0*points_per_deg), 220 - 1.0/(2.0*points_per_deg), 155*points_per_deg)
    signal_binned = np.zeros_like(deg_bin_centers)
    signal_binned_err = np.zeros_like(deg_bin_centers)
    for i in range(len(signal_binned)):
        signal_binned[i] = np.nanmean(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])])
        signal_binned_err[i] = np.nanstd(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])])
        signal_binned_err[i] /= np.sqrt(len(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])]))

    signal_binned /= np.nanmax(signal_binned)
    deg_crossing_idx_98 = np.argwhere(signal_binned < 0.98)[0][0]
    deg_crossing_idx_99 = np.argwhere(signal_binned[1:] < 0.99)[0][0]+1
    deg_crossing_98 = deg_bin_centers[deg_crossing_idx_98]
    deg_crossing_99 = deg_bin_centers[deg_crossing_idx_99]
    plt.figure(figsize=(10, 6))
    plt.xlim(64, 76)
    plt.ylim(0.8, 1.1)
    plt.axvline(x=deg_crossing_99, c="g", ls="--", label="99%% power (deg=%.2f)" % deg_crossing_99)
    plt.axvline(x=deg_crossing_98, c="y", ls="--", label="98%% power (deg=%.2f)" % deg_crossing_98)
    plt.scatter(deg_bin_centers, signal_binned, s=60)
    plt.errorbar(deg_bin_centers, signal_binned, signal_binned_err*10, label="10*err(power)")
    plt.xlabel("Vane angle [Degrees]")
    plt.ylabel("Normalized TOD")
    plt.legend()
    plt.tight_layout()
    plt.title("Power/Angle binned to 0.25 deg. Feed %s" % feed)
    plt.grid(True, which="major", ls="--")
    plt.xticks(range(64,76))
    plt.savefig("plots/binned%d.png" % feed, bbox_inches="tight")
    plt.close()
    
    
for k, feed in enumerate([8,10]):

    clean_idx = ~np.isnan(data[feed])
    signal = data[feed][clean_idx]
    deg = data[0][clean_idx]
    signal = signal[np.argsort(deg)]
    deg = deg[np.argsort(deg)]
    signal = signal[deg > 65]
    deg = deg[deg > 65]

    points_per_deg = 4
    deg_bins = np.linspace(65, 220, 155*points_per_deg + 1)
    deg_bin_centers = np.linspace(65 + 1.0/(2.0*points_per_deg), 220 - 1.0/(2.0*points_per_deg), 155*points_per_deg)
    signal_binned = np.zeros_like(deg_bin_centers)
    signal_binned_err = np.zeros_like(deg_bin_centers)
    for i in range(len(signal_binned)):
        signal_binned[i] = np.nanmean(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])])
        signal_binned_err[i] = np.nanstd(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])])
        signal_binned_err[i] /= np.sqrt(len(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])]))

    signal_binned /= np.nanmax(signal_binned)
    deg_crossing_idx_98 = np.argwhere(signal_binned < 0.98)[0][0]
    deg_crossing_idx_99 = np.argwhere(signal_binned[1:] < 0.99)[0][0]+1
    deg_crossing_98 = deg_bin_centers[deg_crossing_idx_98]
    deg_crossing_99 = deg_bin_centers[deg_crossing_idx_99]
    plt.figure(figsize=(10, 6))
    plt.xlim(64, 76)
    plt.ylim(-0.03, 0.03)
    plt.axvline(x=deg_crossing_99, c="g", ls="--", label="99%% power (deg=%.2f)" % deg_crossing_99)
    plt.axvline(x=deg_crossing_98, c="y", ls="--", label="98%% power (deg=%.2f)" % deg_crossing_98)
    plt.scatter(deg_bin_centers[1:], signal_binned[1:] - signal_binned[:-1], s=60)
    plt.xlabel("Vane angle [Degrees]")
    plt.ylabel("Normalized TOD")
    plt.legend()
    plt.tight_layout()
    plt.title("Power/Angle binned to 0.25 deg. Stepwise diff. Feed %s" % feed)
    plt.grid(True, which="major", ls="--")
    plt.xticks(range(64,76))
    plt.savefig("plots/binned_diff%d.png" % feed, bbox_inches="tight")
    plt.close()


plt.figure(figsize=(12, 8))
for k, feed in enumerate([1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19]):
    clean_idx = ~np.isnan(data[feed])
    signal = data[feed][clean_idx]
    deg = data[0][clean_idx]
    signal = signal[np.argsort(deg)]
    deg = deg[np.argsort(deg)]
    signal = signal[deg > 65]
    deg = deg[deg > 65]

    points_per_deg = 4
    deg_bins = np.linspace(65, 220, 155*points_per_deg + 1)
    deg_bin_centers = np.linspace(65 + 1.0/(2.0*points_per_deg), 220 - 1.0/(2.0*points_per_deg), 155*points_per_deg)
    signal_binned = np.zeros_like(deg_bin_centers)
    signal_binned_std = np.zeros_like(deg_bin_centers)
    for i in range(len(signal_binned)):
        signal_binned[i] = np.nanmean(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])])
        signal_binned_std[i] = np.nanstd(signal[(deg > deg_bins[i])*(deg < deg_bins[i+1])])

    signal_binned /= np.nanmax(signal_binned)
    deg_crossing_idx_98 = np.argwhere(signal_binned < 0.98)[0][0]
    deg_crossing_idx_99 = np.argwhere(signal_binned[1:] < 0.99)[0][0]+1
    deg_crossing_98 = deg_bin_centers[deg_crossing_idx_98]
    deg_crossing_99 = deg_bin_centers[deg_crossing_idx_99]
    # plt.scatter(deg, signal, s=0.2, label="FEED=%d" % feed)
    # plt.xlim(64, 74)
    plt.xlim(60, 140)
    # plt.ylim(-0.03, 0.03)
    plt.ylim(0, 1.1)
    # plt.axvline(x=deg_crossing_99, c="g", ls="--", label="99%% power (deg=%.2f)" % deg_crossing_99)
    # plt.axvline(x=deg_crossing_98, c="y", ls="--", label="98%% power (deg=%.2f)" % deg_crossing_98)
    # plt.axhline(y=0)
    # plt.scatter(deg_bin_centers[1:], signal_binned[1:] - signal_binned[:-1], s=60)
    plt.scatter(deg_bin_centers, signal_binned, s=10, label="feed %s" % feed)
    # plt.errorbar(deg_bin_centers, signal_binned, signal_binned_std, label="std(power)")
plt.xlabel("Vane angle [Degrees]")
plt.ylabel("Normalized TOD")
plt.legend(loc=1)
plt.tight_layout()
plt.title("Power/Angle binned to 0.25 deg.")
plt.savefig("plots/binned_all.png", bbox_inches="tight")
plt.close()
