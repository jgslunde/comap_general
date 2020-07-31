import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import math
from glob import glob
from tqdm import tqdm
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

l1_dir = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro'
#l1_dir = '/Users/ihle/Documents/comap_data'


filenames = glob(l1_dir + '/**/20*-*/*comap-00*.hd5', recursive=True)
# filenames = glob(l1_dir + '/**/2020-01/*comap-00*.hd5', recursive=True)


n_samp_cut = 400

# "hk/antenna0/vane/state",          data%amb_state)                                                                        
#     call read_hdf(file, "hk/antenna0/vane/utc",
def get_tsys_info(filename):
    with h5py.File(filename, mode="r") as my_file:
        obs_id = filename[-29:-22]# + filename[-5:-3]
        #tod = np.array(my_file['spectrometer/tod'][det,sb,freq_hr, n_start:n_cut])
        #t0 = np.array(my_file['spectrometer/MJD'])[0]
        #time = np.array(my_file['spectrometer/MJD'])[n_start:n_cut]
        feat_arr = np.array(my_file[u'/hk/array/frame/features'])
        att = my_file['comap'].attrs
        field = att['source'].decode('utf-8') 
        assert(field[:2] == 'co')
    
        vane = np.array(my_file[u'/hk/antenna0/vane/state'])#position']) #state'])

    vane1 = vane[:n_samp_cut]
    vane2 = vane[-n_samp_cut:]
    feat1 = feat_arr[:n_samp_cut]
    feat2 = feat_arr[-n_samp_cut:]
    
    n_on = [len(np.where(vane1 == 1)[0]), len(np.where(vane2 == 1)[0])] 
    n_stuck = [len(np.where(vane1 == 4)[0]), len(np.where(vane2 == 4)[0])] 
    return n_on, n_stuck, obs_id
    # print(vane == 0)
    #print(obs_id)
    
    # 0 - vane not covering feeds(cold)
    # 1 - vane covering feeds(hot)
    # 2 - vane moving from cold to hot
    # 3 - vane moving from hot to cold
    # 4 - vane stuck
stuck = []
obsids = []
for filename in tqdm(filenames):
    try:
        n_on, n_stuck, obsid = get_tsys_info(filename)
        stuck.append(n_stuck)
        obsids.append(obsid)
    except:
        pass
stuck = np.array(stuck).astype(int)
obsids = np.array(obsids).astype(int)
plt.figure(figsize=(12,6))
plt.scatter(obsids, stuck[:, 0], label='start', s=10)
plt.scatter(obsids + 0.5, stuck[:, 1], label='end', s=10)
plt.ylim(-0.3, 6.3)
plt.xlabel('obsid')
plt.ylabel('# stuck samples')
plt.legend()
plt.tight_layout()
plt.savefig("plots/n_stuck.png", bbox_inches="tight")
plt.close()
plt.clf()