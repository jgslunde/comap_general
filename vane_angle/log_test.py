import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gen_log(x, A, K, B, Q, mu, M, C):
    return A + (K - A)/(C + Q*np.exp(-B*(x-M)))**(1.0/mu)

x = np.linspace(-1.5, 1.5, 1000)
y = gen_log(x, 0, -1, 3, 0.5, 0.5, 0, 1)


y2 = y + np.random.uniform(-1, 1, 1000)

popt, pcov = curve_fit(gen_log, x, y2, p0=[0,-1.5, 2, 0.2, 0.6, 0.2, 1.2])

plt.plot(x, y2)
plt.plot(x, gen_log(x, *popt))
plt.savefig("test.png")
plt.close()

