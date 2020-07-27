# Compiling l2gen
All programs in the comap github must be compiled by the Makefile in the main directory. Compile l2gen (and quiet, which it needs), as
```
make libquiet l2gen
```
Make sure to have enabled a bunch of commandline options and modules (see ~/.bashrc)

# Running l2gen
```
time mpirun -n 1 src/f90/l2gen/l2gen /mn/stornext/d16/cmbco/comap/protodir/param_test_compress.txt
```

```
export OMP_NUM_THREADS=128;time mpirun -n 1 src/f90/l2gen/l2gen /mn/stornext/d16/cmbco/comap/protodir/param_test_uncompressed.txt | tee test_uncompressed_output.txt
```

```
export OMP_NUM_THREADS=16;time mpirun -n 8 comap/src/f90/l2gen/l2gen /mn/stornext/d16/cmbco/comap/protodir/param_jonas_all_co7.txt | tee /mn/stornext/d16/cmbco/comap/jonas/l2gen_co7_log.txt
```

One can specify MPI to run on multiple machines, as:
```
export OMP_NUM_THREADS=16;time mpirun -env I_MPI_FABRICS ofi -machine machinefile.txt -n 24 comap/src/f90/l2gen/l2gen /mn/stornext/d16/cmbco/comap/protodir/param_jonas_all_co7.txt 2>&1| tee /mn/stornext/d16/cmbco/comap/jonas/l2gen_co7_log.txt
```
-n 24 now specifies the combined number of cores over all machines. The file machinefile.txt should look like:
```
owl18:4
owl19:4
owl20:4
owl21:4
owl22:4
owl23:4
```
where 4 specifies the number of cores on that machine.


# Param file
The param file contains the configuration of the l2gen run. The default file can be found in 
```
/mn/stornext/d16/cmbco/comap/protodir/param_default.txt
```
It contains, among other things, the location of the *runlist*.

# Runlist
The runlist specifies which files should be run, how many scans they involve, and some more info about the scans.
A runlist file containing all scans can be found in 
```
/mn/stornext/d16/cmbco/comap/protodir/runlist_default.txt
```
and we can dissect it to get a l2gen on a smaller set of files.

Below is the first lines of runlist_default.txt
The first line specifies the total number of observation targets, like Jupyter, co6 or co7 (two co fields).
For each of the observation fields, the first line specifies a name, and how many obs_ids the field contains.
For each obs_id, the first line contains the actual obs_id, the start and end-time, as well as a filename.
Then, the specific scans follow. The first and last are typically calibration scans.
```
13 
jupiter   199 
  006339   58641.3239005787  58641.3576350383 03 /2019-06/comap-0006339-2019-06-07-074625.hd5 
     00633901  58641.3242592593  58641.3246238426 8192 172.789397  28.632559   0.003674 1 0 0 0 0 0 0 0 0 0  
     00633902  58641.3246354167  58641.3572164352 512 180.888711  30.249045   0.839328 1 0 0 0 0 0 0 0 0 0  
     00633903  58641.3572280093  58641.3575925926 8192 185.353487  28.564557   0.000037 1 0 0 0 0 0 0 0 0 0  
  006407   58645.3104630787  58645.3437449862 03 /2019-06/comap-0006407-2019-06-11-072704.hd5 
     00640701  58645.3108275463  58645.3109490741 8192 172.359286  28.630743   0.001043 1 0 0 1 0 0 0 0 0 0  
     00640702  58645.3109606481  58645.3435648148 512 180.416303  30.285754   0.858595 1 0 0 1 0 0 0 0 0 0  
     00640703  58645.3435763889  58645.3436979167 8192 184.886063  31.665084   0.000053 1 0 0 1 0 0 0 0 0 0  
```


# L2 files
The final location of the l2 files are specified in the param file, and should be a folder in
```
/mn/stornext/d16/cmbco/comap/protodir/level2
```