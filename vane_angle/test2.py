import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
from scipy.optimize import curve_fit


def gen_log(x, A, K, B, mu, Q, M, C):
    return A + (K - A)/(C + Q*np.exp(-B*(x-M)))**(1/mu)

x = np.linspace(60, 220, 1000)

asdf = gen_log(x, 0, 1, 3, 0.5, 0.5, 0, 1)

plt.plot(x, asdf)
plt.show()