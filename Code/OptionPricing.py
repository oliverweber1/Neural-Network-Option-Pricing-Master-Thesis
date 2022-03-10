from StochasticProcesses import *
from MultivariateProcesses import *
from tqdm import tqdm

class AssetModel(StochasticProcess):
    """
    Asset model under risk neutral measure S_t = s0 * exp(r*t - mu*t + X_t)
    for a StochasticProcess (X_t)_t>=0
    Additional drift mu can be used to "center" models (martingale condition for exp(-r*t)*S_t)
    """

    def __init__(self, logPriceProcess, s0=1., mu=0., r=0.05, d=1):
        super().__init__(name=logPriceProcess.name.replace('Process', 'Model'), T=logPriceProcess.T, x0=s0, nSteps=logPriceProcess.nSteps, d=d)
        self.logPriceProcess = logPriceProcess
        if self.d > 1:
            self.mu = np.ones((self.d,1)) * mu if np.isscalar(mu) else mu.reshape((self.d,1))
        else:
            self.mu = mu
        self.r = r

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = self.x0 if x0 is None else x0
        return _x0 * np.exp((self.r - self.mu) * _t + self.logPriceProcess.generateValues(nVals, _t))

    def generatePaths(self, nPaths=1):
        return self.x0 * np.exp((self.r - self.mu) * self.timePoints + self.logPriceProcess.generatePaths(nPaths).T).T

    def OptionPriceMC(self, payoffFunc, assetVal=None, expiry=None, nSim=1000000):
        # calculates price of an option given by its payoff function, the current underlying value and time to maturity
        T = expiry or self.T
        s = self.x0 if assetVal is None else assetVal
        if self.d > 1:
            s = s.reshape((self.d, 1))
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


class MultiAssetModel(AssetModel, MultivariateProcess):
    """
    Slight extension of AssetModel to make it work with multivariate processes
    """

    def generatePath(self):
        return (self.x0 * np.exp((self.r - self.mu) * self.timePoints + self.logPriceProcess.generatePath().T)).T

    def plotPath(self, figSize=(10,5)):
        return MultivariateProcess.plotPath(self, figSize)

    def OptionPriceRangeMC(self, payoffFunc, assetStartVals, expiry=None, nSim=1000000):
        """
        Divides option price generation into prices with 10k simulations and combines them
        in order to free up memory (required in higher dimensions)
        """
        batchSize = 10000
        batches, rest = divmod(nSim, batchSize)
        if batches == 0: # less than 10k simulations
            return AssetModel.OptionPriceRangeMC(self, payoffFunc, assetStartVals, expiry, rest)
        batchMean = np.zeros(len(assetStartVals))
        for _ in tqdm(range(batches)):
            batchMean += AssetModel.OptionPriceRangeMC(self, payoffFunc, assetStartVals, expiry, batchSize)
        batchMean /= batches
        if rest > 0: # take weighted mean with rest of simulations and the 10k batch mean
            restVals = AssetModel.OptionPriceRangeMC(self, payoffFunc, assetStartVals, expiry, rest)
            batchMean = (batches * batchSize * batchMean + rest * restVals) / nSim
        return batchMean


class MultiAssetMertonModel(MultiAssetModel):
    """
    Extension of Merton Jump Diffusion Model to muldi-dimensional markets
    """

    def __init__(self, d, lam=3, mu_j=None, sig_j=None, Corr_j=None, sig_bm=None, Corr_bm=None, r=0.05, s0=1., T=1., nSteps=10000):
        _mu_j = np.zeros((d, 1)) if mu_j is None else np.reshape(mu_j, (d, 1))
        _sig_j = np.ones((d,1)) if sig_j is None else np.reshape(sig_j, (d,1))
        _sig_bm = np.ones((d,1)) if sig_bm is None else np.reshape(sig_bm, (d,1))
        JumpPart = MultivariateCompoundPoissonProcess(d=d, lam=lam, T=T, mu=mu_j, sigma=sig_j, Corr=Corr_j, nSteps=nSteps)
        BrownianPart = CorrelatedBrownianMotion(d=d, T=T, sigma=sig_bm, Corr=Corr_bm, nSteps=nSteps)
        logPriceProcess = MultivariateJumpDiffusionProcess(BrownianPart=BrownianPart, JumpPart=JumpPart)
        expMomentJump = np.exp(_mu_j + _sig_j ** 2 / 2)
        mu = _sig_bm ** 2 / 2 + lam * (expMomentJump - 1) # martingale correction
        super().__init__(logPriceProcess, s0, mu, r, d)
        self.name = 'Multivariate Merton Jump Diffusion Model'

class MultiAssetNIGModel(MultiAssetModel):
    """
    Extension of NIG Model to multi-dimensional markets
    """

    def __init__(self, d, kappa=1., theta=None, sigma=None, Corr=None, r=0.05, s0=1., T=1, nSteps=10000):
        _theta = np.zeros((d, 1)) if theta is None else np.reshape(theta, (d, 1))
        _sigma = np.ones((d,1)) if sigma is None else np.reshape(sigma, (d,1))
        logPriceProcess = MultivariateNIGProcess(d=d, T=T, nSteps=nSteps, kappa=kappa, theta=theta, sigma=sigma, Corr=Corr)
        mu = (1 - np.sqrt(1 - 2 * kappa * _theta - kappa * _sigma ** 2)) / kappa # martingale correction
        super().__init__(logPriceProcess, s0, mu, r, d)
        self.name = 'Multivariate Normal Inverse Gaussian Model'
