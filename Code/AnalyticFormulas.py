import scipy.stats as sp
import numpy as np

def blackScholesPrice(s, r, sigma, T, K, put=False):
    d1 = (np.log(s / K) + (r + sigma ** 2) * T) / (sigma * np.sqrt(T)) * (1 -2 * put)
    d2 = d1 - sigma * np.sqrt(T) * (1 -2 * put)
    return ((s * sp.norm.cdf(d1, 0., 1.) - K * np.exp(-r * T) * sp.norm.cdf(d2, 0., 1.))) * (1 -2 * put)
