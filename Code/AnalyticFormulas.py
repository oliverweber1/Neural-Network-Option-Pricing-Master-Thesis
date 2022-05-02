import scipy.stats as sp
import numpy as np
from math import factorial

def blackScholesPrice(s, r, sigma, T, K, put=False):
    d1 = (np.log(s / K) + (r + sigma ** 2) * T) / (sigma * np.sqrt(T)) * (1 -2 * put)
    d2 = d1 - sigma * np.sqrt(T) * (1 -2 * put)
    return ((s * sp.norm.cdf(d1, 0., 1.) - K * np.exp(-r * T) * sp.norm.cdf(d2, 0., 1.))) * (1 -2 * put)


def MertonJumpDiffusionPrice(s, r, sigma, T, K, mu_j, sig_j, lam, nSum=100, put=False):
    """
    Analytic Formula for Merton Price as infinite sum of BS Prices, see Tankov 10.1
    """
    price = 0.
    for n in range(nSum):
        sig_n = np.sqrt(sigma ** 2 + n * sig_j ** 2 / T)
        s_n = s * np.exp(lam * T + n * mu_j - lam * T * np.exp(mu_j + sig_j ** 2 / 2) + n * sig_j ** 2 / 2)
        price += np.exp(-lam * T) * (lam * T) ** n / factorial(n) * blackScholesPrice(s_n, r, sig_n, T, K, put)
    return price