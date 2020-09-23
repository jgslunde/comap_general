import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import copy
import scipy.stats

try:
    filename = sys.argv[1]
except IndexError:
    print('Missing filename!')
    print('Usage: python diag_l2.py filename')
    sys.exit(1)

with h5py.File(filename, mode="r") as my_file:
    try:
        scan_id = my_file['scanid'][()]
    except:
        scan_id = filename[-12:-3]# + filename[-5:-3]
    tod_ind = np.array(my_file['tod'][:])
    n_det_ind, n_sb, n_freq, n_samp = tod_ind.shape
    mask_ind = my_file['freqmask'][:]
    mask_full_ind = my_file['freqmask_full'][:]
    pixels = np.array(my_file['pixels'][:]) - 1 
    pix2ind = my_file['pix2ind'][:]
    try:
        acc_ind = np.array(my_file['acceptrate'])
    except KeyError:
        acc_ind = np.zeros_like(tod_ind[:,:,0,0])
        print("Found no acceptrate")
    time = np.array(my_file['time'])
    try:
        pca = np.array(my_file['pca_comp'])
        eigv = np.array(my_file['pca_eigv'])
        ampl_ind = np.array(my_file['pca_ampl'])
    except KeyError:
        pca = np.zeros((4, 10000))
        eigv = np.zeros(0)
        ampl_ind = np.zeros((4, n_det_ind, n_sb, 1024))
        print('Found no pca comps')
    try:
        tsys_ind = np.array(my_file['Tsys_lowres'])
    except KeyError:
        tsys_ind = np.zeros_like(tod_ind[:,:,:]) + 40
        print("Found no tsys")



time = (time - time[0]) * (24 * 60)  # minutes


n_freq_hr = len(mask_full_ind[0,0])
n_det = np.max(pixels) + 1 
# print(n_det)

## transform to full arrays with all pixels
tod = np.zeros((n_det, n_sb, n_freq, n_samp))
mask = np.zeros((n_det, n_sb, n_freq))
mask_full = np.zeros((n_det, n_sb, n_freq_hr))
acc = np.zeros((n_det, n_sb))
ampl = np.zeros((4, n_det, n_sb, n_freq_hr))
tsys = np.zeros((n_det, n_sb, n_freq))

# print(ampl_ind.shape)
# print(ampl[:, pixels, :, :].shape)

tod[pixels] = tod_ind
mask[pixels] = mask_ind
mask_full[pixels] = mask_full_ind
acc[pixels] = acc_ind
ampl[:, pixels, :, :] = ampl_ind
tsys[pixels] = tsys_ind


acc = acc.flatten()
mask_full = mask_full.reshape((n_det, n_sb, n_freq, 16)).sum(3)
# tsys[:] = 1.0
#tsys = tsys[:, :, :, 1].reshape((n_det, n_sb, n_freq, 16)).mean(3)
# n_det = 19
tod = tod[:, :, :, :] * mask[:, :, :, None]
tod_hist = copy.deepcopy(tod)
tod[:, (0, 2)] = tod[:, (0, 2), ::-1]

tod_flat = tod.reshape((n_det * n_sb * n_freq, n_samp))
corr = np.corrcoef(tod_flat)
fig = plt.figure()
ax = fig.add_subplot(111)
# im = ax.imshow(np.log10(np.abs(corr)), vmin=-3, vmax=-1,
#                extent=(0.5, n_det + 0.5, n_det + 0.5, 0.5))
# cbar = fig.colorbar(im, ticks=[-1, -2, -3])
# cbar.ax.set_yticklabels(['0.1', '0.01', '0.001'])
vmax = 0.1
im = ax.imshow(corr, vmin=-vmax, vmax=vmax,  #vmin=-1, vmax=1,#vmin=-0.1, vmax=0.1,
                extent=(0.5, n_det + 0.5, n_det + 0.5, 0.5))
cbar = fig.colorbar(im)
cbar.set_label('Correlation')
new_tick_locations = np.array(range(n_det)) + 1
ax.set_xticks(new_tick_locations)
ax.set_yticks(new_tick_locations)
# ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
xl = np.linspace(0.5, n_det + 0.5, n_det * 1 + 1)
ax.vlines(xl, ymin=0.5, ymax=n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
ax.hlines(xl, xmin=0.5, xmax=n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
# ax.hlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), xmin=0.5, xmax=n_det + 0.5, linestyle='--', color='k', lw=0.01, alpha=0.1)
for i in range(n_det):
    plt.text(xl[i] + 0.7, xl[i] + 0.2, str(i+1), rotation=0, verticalalignment='center', fontsize=2)
ax.set_xlabel('Feed')
ax.set_ylabel('Feed')
ax.set_title('Scan: ' + str(scan_id))
save_string = 'corr_%08i.png' % (int(scan_id))
plt.savefig(save_string, bbox_inches='tight', dpi=800)

dt = (time[1] - time[0]) * 60  # seconds
radiometer = 1 / np.sqrt(31.25 * 10 ** 6 * dt)  # * 1.03

# print(tsys)
# for i in range(19): ###########################
tsys[np.where(tsys == 0.0)] = np.inf
# tsys = tsys * np.sqrt(mask_full)
# i = 10
tod2 = (tod_hist / (tsys[:, :, :, None] * radiometer) * np.sqrt(mask_full[:, :, :, None] / 16)).flatten()#tod.flatten() / radiometer#(tod / radiometer * np.sqrt(mask_full[:, :, :, None] / 16)).flatten()
tod2 = tod2[np.where(np.abs(tod2) > 0)]
tod2 = tod2[np.where(np.abs(tod2) < 20)]
std = np.nanstd(tod2)
print(std)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
x = np.linspace(-4 * std, 4 * std, 300)
ax1.plot(x, scipy.stats.norm.pdf(x), 'g', lw=2, label=r'$\mathcal{N}(0, 1)$', zorder=1)
ax1.plot(x, scipy.stats.norm.pdf(
    x, scale=std), 'r', lw=2,
    label=r'$\mathcal{N}(0, \sigma_\mathrm{samp})$',
    zorder=2)
ax1.hist(tod2, bins=x, density=True, label='All samples, ', alpha=0.8, zorder=3)
# plt.yscale('log')
# ax1.legend()
ax1.set_xlim(x[0], x[-1])
ax1.set_xlabel(r'$x \cdot \sqrt{B\tau}$')
ax1.set_ylabel(r'$p(x\cdot \sqrt{B\tau})$')
ax1.text(-3.95, .395, 'Scan: ' + str(scan_id), fontsize=10) # ', Feed: ' + str(i+1)

ax2 = fig.add_subplot(122)
x = np.linspace(-6 * std, 6 * std, 400)
ax2.plot(x, scipy.stats.norm.pdf(x), 'g', lw=2, label=r'$\mathcal{N}(0, 1)$', zorder=1)
ax2.plot(x, scipy.stats.norm.pdf(
    x, scale=std), 'r', lw=2,
    label=r'$\mathcal{N}(0, \sigma_\mathrm{samp})$',
    zorder=2)
ax2.hist(tod2, bins=x, density=True, label='All samples, ', alpha=0.8, zorder=3)
ax2.set_yscale('log')
ax2.legend()
ax2.set_xlim(x[0], x[-1])
ax2.set_ylim(1e-6, 1e0)
ax2.set_xlabel(r'$x \cdot \sqrt{B\tau}$')
# ax2.set_ylabel(r'$p(x\cdot \sqrt{B\tau})$')
# ax2.text(-3.9, .36, 'Scan: ' + str(scan_id), fontsize=15)

save_string = 'hist_%08i.png' % (int(scan_id))  # , i+1)
plt.savefig(save_string, bbox_inches='tight')


# a = np.abs(ampl).mean((1, 2, 3))
# a2 = np.abs(ampl ** 2).mean((1, 2, 3))
# # print(a)
# i_max = np.unravel_index(a.argmax(), a.shape)[0]  # meanamp.index(max(meanamp))
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# # print(i_max)
# var_exp = a2[i_max] * (pca[i_max]).std() ** 2 / radiometer ** 2 
# ax1.plot(time, pca[i_max], label='PCA common mode, ' + str(scan_id))
# ax1.legend()
# ax1.set_xlim(0, 1)
# # ax1.set_xlabel('time [m]')
# # plt.ylabel(r'TOD')
# # plt.savefig('pca_1min.pdf', bbox_inches='tight')
# ax2 = fig.add_subplot(212)
# ax2.plot(time, pca[i_max], label='Avg std explained: %g' % np.sqrt(var_exp))   # , label='PCA common mode')
# ax2.legend()
# ax2.set_xlim(0, time[-1])
# ax2.set_xlabel('time [m]')
# # plt.ylabel(r'TOD')
# # plt.savefig('pca_fullscan.pdf', bbox_inches='tight')

# save_string = 'pca_%08i.pdf' % (int(scan_id))
# plt.savefig(save_string, bbox_inches='tight')


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(211)

n = len(acc)
# for j in range(n):
#     print(np.array(file_list[i]['freqmask_full']).flatten()[j])
x = np.linspace(0.5, n_det + 0.5, len(acc) + 1)
plotter = np.zeros((len(acc) + 1))
plotter[:-1] = acc  ## is this wrong???
ax.step(x, plotter, where='post', label='Mean acceptrate: ' + str(acc.mean() * 20 / 19))##############, width=0.3)

ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
ax.vlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.2, alpha=0.2)
new_tick_locations = np.array(range(n_det)) + 1
ax.set_xticks(new_tick_locations)
ax.set_xlim(0.5, n_det + 0.48)
ax.set_ylim(0.75, 1)
# ax.text(1.0, 0.81, 'Mean acceptrate' + str(acc.mean()), fontsize=8)
# ax.set_xlabel('Detector')
ax.set_ylabel('Acceptance rate')
ax.legend(loc=3)
# save_string = 'acc_%08i.pdf' % (int(scan_id))
# plt.savefig(save_string, bbox_inches='tight')


tod = tod_hist[:, :, :, :]
tod = tod * np.sqrt(mask_full[:, :, :, None] / 16)
# tod[:, (0, 2)] = tod[:, (0, 2), ::-1]

std = tod.std(3)   ##########################################3
std[np.where(std != 0)] = std[np.where(std != 0)] / tsys[np.where(std != 0)]
mask = np.zeros_like(std)
mask[np.where(std != 0)] = 1.0
mask = mask.reshape(n_det, n_sb, n_freq)

mean_std = std.sum(2)
# print(mean_std)
mean_std[np.where(mean_std > 0)] = mean_std[np.where(mean_std > 0)] / mask.sum(2)[np.where(mean_std > 0)]
mean_std = mean_std.flatten()
mean_std[np.where(mean_std == 0)] = 1e3
x = np.linspace(0.5, n_det + 0.5, len(mean_std) + 1)
# fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(212)
plotter = np.zeros((len(mean_std) + 1))
# print(plotter.shape)
plotter[:-1] = mean_std
ax.step(x, plotter / radiometer, where='post', label=str(scan_id))
# ax.step(x, 0*x + 1)
new_tick_locations = np.array(range(n_det)) + 1
ax.set_xticks(new_tick_locations)
ax.set_xlim(0.5, n_det + 0.5)
# ax.xaxis.grid((0.5, n_det + 0.5, 20))
# print(np.linspace(0.5, n_det + 0.5, 20))
ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=1, ymax=2, linestyle='--', color='k', lw=0.6, alpha=0.2)
ax.vlines(np.linspace(0.5, n_det + 0.5, n_det * n_sb + 1), ymin=1, ymax=2, linestyle='--', color='k', lw=0.2, alpha=0.2)
# ax.axvline(x[5], ymin=0, ymax=2, color='k', lw=6)
for i in range(len(mean_std)):
    if mean_std[i] == 1e3:
        ax.axvspan(x[i], x[i + 1], alpha=1, color='k', zorder=5)
# plt.grid()
ax.set_ylim(1, 1.4)
ax.set_xlim(0.5, n_det + 0.48)
ax.legend(loc=0)
# ax.set_xlabel('Feed')
ax.set_ylabel(r'$\langle\sigma_\mathrm{TOD}\rangle \cdot \sqrt{B\tau}$')
save_string = 'acc_var_%08i.png' % (int(scan_id))
plt.savefig(save_string, bbox_inches='tight')

# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(313)
# n_det = 19
# scanid = 2062
a = np.abs(ampl).mean((1, 2, 3))
a2 = np.abs(ampl ** 2).mean((1, 2, 3))
fig = plt.figure(figsize=(6, 12))
# ax = fig.add_subplot(611)
# radiometer *= 40
for i in range(3):
    subplot = str(611 + 2 * i)
    # i = i + 3
    var_exp = a2[i] * (pca[i]).std() ** 2 / radiometer ** 2 
    ax2 = fig.add_subplot(subplot)
    ax2.plot(time, pca[i], label=str(scan_id) + ", PCA comp. " + str(i+1))   # , label='PCA common mode')
    ax2.legend()
    ax2.set_xlim(0, time[-1])
    ax2.set_xlabel('time [m]')
    # i = i - 3
    subplot = str(611 + 2 * i + 1)
    # i = i + 3
    ax = fig.add_subplot(subplot)
    #my_file = h5py.File(filename, mode="r")
    acc = ampl[i].flatten() #np.array(my_file['pca_ampl'][i]).flatten()
    n = len(acc)
    # print(n)
    n_dec1 = 16
    acc = np.abs(acc.reshape((n // n_dec1, n_dec1)).mean(1))
    n_dec2 = 64
    acc = np.abs(acc.reshape((n // (n_dec1 * n_dec2), n_dec2))).mean(1)
    # acc = acc.reshape((n // n_dec, n_dec)).mean(1)
    n = len(acc)
    # print(n)
    acc = 100 * np.sqrt(acc ** 2 * (pca[i]).std() ** 2 / radiometer ** 2)
    # print(acc)
    # for j in range(n):
    #     print(np.array(file_list[i]['freqmask_full']).flatten()[j])
    x = np.linspace(0.5, n_det + 0.5, len(acc) + 1)
    plotter = np.zeros((len(acc) + 1))
    plotter[:-1] = acc
    ax.step(x, plotter, label='Avg std explained: %.1f %%' % (acc.mean()), where='post')
    # x = np.linspace(0.5, n_det + 0.5, len(acc))
    # ax.plot(x, acc, label="co2_" + str(scanid) + "_0" + str(i+1))#, width=0.3)

    ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=-1, ymax=100, linestyle='--', color='k', lw=0.6, alpha=0.2)
    ax.vlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), ymin=-1, ymax=100, linestyle='--', color='k', lw=0.2, alpha=0.2)
    new_tick_locations = np.array(range(n_det)) + 1
    ax.set_xticks(new_tick_locations)
    ax.set_xlim(0.5, n_det + 0.48)
    # ax.set_ylim(-0.05, 0.05)
    ax.set_ylim(0, 60)
    ax.set_xlabel('Feed')
    ax.set_ylabel(r'% of $1 / \sqrt{B\tau}$')
    ax.legend(loc=1)
    # i = i - 3
save_string = 'pca_ampl_%08i.png' % (int(scan_id))
plt.savefig(save_string, bbox_inches='tight')
