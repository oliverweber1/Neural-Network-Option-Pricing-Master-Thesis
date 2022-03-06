from StochasticProcesses import *

class AssetModel(StochasticProcess):
    """
    Asset model under risk neutral measure S_t = s0 * exp(r*t - mu*t + X_t)
    for a StochasticProcess (X_t)_t>=0
    Additional drift mu can be used to "center" models (martingale condition for exp(-r*t)*S_t)
    """

    def __init__(self, logPriceProcess, s0=1., mu=0., r=0.05):
        super().__init__(name=logPriceProcess.name.replace('Process', 'Model'), T=logPriceProcess.T, x0=s0, nSteps=logPriceProcess.nSteps)
        self.logPriceProcess = logPriceProcess
        self.mu = mu
        self.r = r

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        return _x0 * np.exp((self.r - self.mu) * _t + self.logPriceProcess.generateValues(nVals, _t))

    def generatePaths(self, nPaths=1):
        return self.x0 * np.exp((self.r - self.mu) * self.timePoints + self.logPriceProcess.generatePaths(nPaths).T).T

    def OptionPriceMC(self, payoffFunc, assetVal=None, expiry=None, nSim=1000000):
        # calculates price of an option given by its payoff function, the current underlying value and time to maturity
        T = expiry or self.T
        s = assetVal or self.x0
        assert T <= self.T, 'Time to maturity is not covered by the asset model'
        payoff = payoffFunc(self.generateValues(nSim, T, s))
        return np.exp(-self.r * T) * payoff.mean()

    def OptionPriceRangeMC(self, payoffFunc, assetStartVals, expiry=None, nSim=1000000):
        # calclates option prices for an array of underlying values and fixed payoff
        T = expiry or self.T
        assert T <= self.T, 'Time to maturity is not covered by the asset model'
        assetEndVals = self.generateValues(nSim, T, 1.)
        return np.array([np.exp(-self.r * T) * payoffFunc(s * assetEndVals).mean() for s in assetStartVals])

    def OptionPricePayOffRangeMC(self, payoffFuncs, assetVal=None, expiry=None, nSim=1000000):
        # calculates option prices for an array of payoff functions and fixed underlying value
        T = expiry or self.T
        assert T <= self.T, 'Time to maturity is not covered by the asset model'
        s = assetVal or self.x0
        assetEndVals = self.generateValues(nSim, T, s)
        return np.array([np.exp(-self.r * T) * payoff(assetEndVals).mean() for payoff in payoffFuncs])


class MertonModel(AssetModel):
    """
    Jump diffusion model that assumes normally distributed jumps
    """

    def __init__(self, lam=3, mu_j=0., sig_j=0.1, sigma=0.2, r=0.05, s0=1., T=1., nSteps=10000):
        jumpRV = sp.norm(loc=mu_j, scale=sig_j)
        expMomentJump = np.exp(mu_j + sig_j ** 2 / 2)
        mu = sigma ** 2 / 2 + lam * (expMomentJump - 1) # martingale correction
        BrownianPart = BrownianMotionWithDrift(T=T, nSteps=nSteps, sigma=sigma)
        JumpPart = CompoundPoissonProcess(lam=lam, T=T, nSteps=nSteps, jumpSizeRV=jumpRV)
        logPrice = JumpDiffusionProcess(BrownianPart, JumpPart)
        super().__init__(logPrice, s0=s0, mu=mu, r=r)
        self.name = 'Merton Model'

class NIGModel(AssetModel):
    """
    NIG model
    """

    def __init__(self, theta=0.1, sigma=0.3, kappa=0.2, r=0.05, s0=1., T=1, nSteps=10000):
        logPrice = NIGProcess(T=T, nSteps=nSteps, theta=theta, sigma=sigma, kappa=kappa)
        mu = (1 - np.sqrt(1 - 2 * kappa * theta - kappa * sigma ** 2)) / kappa
        super().__init__(logPrice, s0=s0, mu=mu, r=r)
        self.name = 'NIG Model'

class BlackScholesModel(AssetModel):
    """
    Black Scholes model (only to compare to analytical model)
    """

    def __init__(self, sigma=0.2, r=0.05, s0=1., T=1., nSteps=10000):
        super().__init__(BrownianMotionWithDrift(T=T, nSteps=nSteps, sigma=sigma), s0=s0, mu=sigma**2/2, r=r)
        self.name = 'Black Scholes Model'



