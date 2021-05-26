import numpy as np
import numpy.ma as ma
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
import h5py

Ntod = 10000
basis_size = 80
Nbasis = Ntod//basis_size

data = np.ones(Ntod)
col_idx = np.arange(Ntod, dtype=int)
row_idx = np.arange(Ntod, dtype=int)//basis_size

F = csc_matrix((data, (col_idx, row_idx)), shape=(Ntod, Nbasis))

Nsidemap = 20
Nmap = Nsidemap**2

# x_idx = (np.arange(Ntod)//Nsidemap)%Nsidemap
# y_idx = np.arange(Ntod)%Nsidemap
filename = 'co7_001495506.h5'
    
with h5py.File(filename, mode="r") as my_file:
    point = np.array(my_file['point_cel']).astype(float)

def find_index(arr, n, K):
     
    # Traverse the array
    for i in range(n):
         
        # If K is found
        if arr[i] == K:
            return i
             
        # If arr[i] exceeds K
        elif arr[i] > K:
            return i
             
    # If all array elements are smaller
    return n
print(point[0, :, 1])

ra_bins = np.linspace(168, 172, Nsidemap+1)
dec_bins = np.linspace(52, 53.5, Nsidemap+1)

ra_inds = np.array([find_index(ra_bins, Nsidemap, ra) for ra in point[0, :Ntod, 0]])
dec_inds = np.array([find_index(dec_bins, Nsidemap, dec) for dec in point[0, :Ntod, 1]])
# x_idx+Nsidemap*y_idx
# P = csc_matrix((np.ones(Ntod), (np.arange(Ntod, dtype=int), x_idx+Nsidemap*y_idx)), shape=(Ntod, Nmap))
P = csc_matrix((np.ones(Ntod), (np.arange(Ntod, dtype=int), ra_inds + Nsidemap*dec_inds)), shape=(Ntod, Nmap))

Corr_wn_inv = scipy.sparse.diags(np.ones(Ntod))

y = np.random.normal(0, 1, Ntod)

PT = csc_matrix(P.T)
FT = csc_matrix(F.T)
inv_PT_C_P = scipy.sparse.diags( 1.0/(PT.dot(Corr_wn_inv).dot(P)).diagonal())
P_inv_PT_C_P = P.dot(inv_PT_C_P)
FT_C_F = FT.dot(Corr_wn_inv).dot(F)
FT_C_P_inv_PT_C_P = FT.dot(Corr_wn_inv.dot(P_inv_PT_C_P))
PT_C_F = PT.dot(Corr_wn_inv).dot(F)

def LHS(a):
    a1 = FT_C_F.dot(a)
    a2 = PT_C_F.dot(a)
    a3 = FT_C_P_inv_PT_C_P.dot(a2)
    return a1 - a3

A = LinearOperator((Nbasis,Nbasis), matvec=LHS)
b = F.T.dot(Corr_wn_inv).dot(y) - FT_C_P_inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y))

def solve_cg(A, b):
    num_iter = 0
    x0 = np.zeros(Nbasis)
    a, info = scipy.sparse.linalg.cg(A, b, x0=x0)
    return a, info

a, info = solve_cg(A, b)
map_nw = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y))
map_destripe = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y-F.dot(a)))
template_map = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(F.dot(a)))
hitmap = PT.dot(P).diagonal()

print(ma.corrcoef([ma.masked_invalid(map_nw), ma.masked_invalid(map_destripe), ma.masked_invalid(template_map)]))

map_destripe = map_destripe.reshape((Nsidemap, Nsidemap))
map_nw = map_nw.reshape((Nsidemap, Nsidemap))
template_map = template_map.reshape((Nsidemap, Nsidemap))
vmax = 0.5
plt.figure()
plt.title('binned')
plt.imshow(map_nw, interpolation='none', vmin=-vmax, vmax=vmax)
plt.savefig('binned.png')
plt.figure()
plt.title('destriped')
plt.imshow(map_destripe, interpolation='none', vmin=-vmax, vmax=vmax)
plt.savefig('destriped.png')
plt.figure()
plt.title('template')
plt.imshow(template_map, interpolation='none', vmin=-vmax, vmax=vmax)
plt.savefig('template.png')
plt.show()

