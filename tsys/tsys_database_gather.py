import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir
from os.path import isfile, join
from tqdm import tqdm, trange
import matplotlib
import time
from tqdm import trange
import multiprocessing as mp
import sys

if __name__ == "__main__":
    inpath = sys.argv[1]  # Path to the obsid hdf5 files.
    outfilename = sys.argv[2]  # Path and name of the database to be written.
    
    filenames = []
    obsids = []
    for f in listdir(inpath):
        if isfile(join(inpath, f)):
            if f[-4:] == ".hd5" or f[-3:] == ".h5":
                filenames.append(join(inpath, f))
                obsids.append(f.split(".")[0])                

    Nfiles = len(filenames)
    print(f"Found {Nfiles} hdf5 files.")

    with h5py.File(outfilename, "w") as outfile:
        for i in trange(Nfiles):
            filename = filenames[i]
            obsid = obsids[i]
            with h5py.File(filename, "r") as infile:
                Thot = infile["Thot"][()]
                Phot = infile["Phot"][()]
                points_used_Thot = infile["points_used_Thot"][()]
                points_used_Phot = infile["points_used_Phot"][()]
                tsys = infile["Tsys_obsidmean"][()]
                successful = infile["successful"][()]
                calib_times = infile["calib_times"][()]
            datagroup = "obsid/" + obsid + "/"
            outfile.create_dataset(datagroup + "Thot", data=Thot)
            outfile.create_dataset(datagroup + "Phot", data=Phot)
            outfile.create_dataset(datagroup + "points_used_Thot", data=points_used_Thot)
            outfile.create_dataset(datagroup + "points_used_Phot", data=points_used_Phot)
            outfile.create_dataset(datagroup + "Tsys_obsidmean", data=tsys)
            outfile.create_dataset(datagroup + "successful", data=successful)
            outfile.create_dataset(datagroup + "calib_times", data=calib_times)