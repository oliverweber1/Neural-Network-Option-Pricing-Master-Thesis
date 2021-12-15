from StochasticProcesses import *

class AssetModel(StochasticProcess):
    """
    Asset model under risk neutral measure S_t = s0 * exp(r*t - mu*t + X_t)
    for a StochasticProcess (X_t)_t>=0
    Additional drift mu can be used for to "center" models (martingale condition for exp(-r*t)*S_t)
    """

    def __init__(self, logPriceProcess, s0=1., mu=0., r=0.05):
        super().__init__(name=logPriceProcess.name.replace('Process', 'Model'), T=logPriceProcess.T, x0=s0, nSteps=logPriceProcess.nSteps)
        self.logPriceProcess = logPriceProcess
        self.mu = mu
        self.r = r

    def generateValues(self, t=None, nVals=1):
        _t = t or self.T
        return self.x0 * np.exp((self.r - self.mu) * _t + self.logPriceProcess.generateValues(_t, nVals))

    def generatePaths(self, nPaths=1):
        return self.x0 * np.exp((self.r - self.mu) * self.timePoints + self.logPriceProcess.generatePaths(nPaths).T).T


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


