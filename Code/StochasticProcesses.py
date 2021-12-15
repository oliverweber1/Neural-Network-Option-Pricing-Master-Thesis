import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt

class StochasticProcess:

    def __init__(self, name, T=1, x0=0., nSteps=10000):
        self.name = name
        self.T = T
        self.x0 = x0
        self.nSteps = nSteps
        self.timePoints = np.linspace(0, T, nSteps)

    def generatePaths(self, nPaths=1):
        raise NotImplementedError('Must override generatePaths')

    def generateValues(self, nVals=1, t=None, x0=None):
        raise NotImplementedError('Must override generateValues')

    def plotPaths(self, figSize=(10,5), nPaths=1):
        plt.figure(figsize=figSize)
        paths = self.generatePaths(nPaths)
        plt.plot(paths)
        plt.ylabel('$S_t$')
        plt.title('{} sample path{} of a {} '.format(nPaths, '' if nPaths == 1 else 's', self.name))


class BrownianMotion(StochasticProcess):

    def __init__(self, T=1, nSteps=10000):
        super().__init__(name='Standard Brownian Motion', T=T, x0=0., nSteps=nSteps)

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        return _x0 + sp.norm.rvs(scale=np.sqrt(_t), size=nVals)

    def generatePaths(self, nPaths=1):
        dt = self.T / (self.nSteps - 1)
        increments = np.append(np.ones((nPaths,1)) * self.x0, sp.norm.rvs(scale=np.sqrt(dt), size=(nPaths,self.nSteps - 1)), axis=1)
        return pd.DataFrame(np.cumsum(increments,axis=1).T, index=self.timePoints)


class BrownianMotionWithDrift(StochasticProcess):

    def __init__(self, T=1, x0=0, nSteps=10000, mu=0., sigma=1.):
        super().__init__(name='Brownian Motion with Drift', T=T, x0=x0, nSteps=nSteps)
        self.mu = mu
        self.sigma = sigma
        self.brownianDriver = BrownianMotion(T=T, nSteps=self.nSteps)

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        return self.x0 + self.mu * _t + self.sigma * self.brownianDriver.generateValues(nVals, _t)

    def generatePaths(self, nPaths=1):
        return (self.x0 + self.mu * self.timePoints + self.sigma * self.brownianDriver.generatePaths(nPaths).T).T


class PoissonProcess(StochasticProcess):

    def __init__(self, lam, T=1, x0=0., nSteps=10000):
        super().__init__(name='Poisson Process', T=T, x0=x0, nSteps=nSteps)
        self.lam = lam

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        return _x0 + sp.poisson.rvs(mu=self.lam * _t, size=nVals)

    def generatePaths(self, nPaths=1):
        paths = pd.DataFrame(np.ones((len(self.timePoints), nPaths)) * self.x0, index=self.timePoints)
        numberOfJumps = sp.poisson.rvs(mu=self.lam*self.T, size=nPaths)
        jumpTimes = np.array([sp.uniform.rvs(scale=self.T, size=n) for n in numberOfJumps], dtype=object)
        for i, jumps in enumerate(jumpTimes):
            for jump in jumps:
                paths[i][paths.index > jump] += 1
        return paths


class CompoundPoissonProcess(StochasticProcess):

    def __init__(self, lam, T=1, x0=0., nSteps=10000, jumpSizeRV=sp.norm):
        super().__init__(name='Compound Poisson Process', T=T, x0=x0, nSteps=nSteps)
        self.lam = lam
        self.jumpSizeRV = jumpSizeRV

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        numOfJumps = sp.poisson.rvs(mu=self.lam*_t, size=nVals)
        maxJumps = np.max(numOfJumps)
        jumpSizes = self.jumpSizeRV.rvs(size=(nVals, maxJumps))
        jumpSizes[numOfJumps[:,None] <= np.arange(jumpSizes.shape[1])] = 0
        return (_x0 + jumpSizes.sum(axis=1))

    def generatePaths(self, nPaths=1):
        paths = pd.DataFrame(np.ones((len(self.timePoints), nPaths)) * self.x0, index=self.timePoints)
        numberOfJumps = sp.poisson.rvs(mu=self.lam*self.T, size=nPaths)
        jumpTimes = np.array([sp.uniform.rvs(scale=self.T, size=n) for n in numberOfJumps], dtype=object)
        for i, jumps in enumerate(jumpTimes):
            for jump in jumps:
                paths[i][paths.index > jump] += self.jumpSizeRV.rvs()
        return paths


class JumpDiffusionProcess(StochasticProcess):

    def __init__(self, BrownianPart, JumpPart, x0=0., mu=0.):
        assert (BrownianPart.timePoints == JumpPart.timePoints).all(), 'Time grids of Brownian part and jump part do not coincide!'
        super().__init__(name='Jump Diffusion Process', T=BrownianPart.T, x0=x0, nSteps=BrownianPart.nSteps)
        self.BrownianPart = BrownianPart
        self.JumpPart = JumpPart
        self.mu = mu

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        return _x0 + _t * self.mu + self.BrownianPart.generateValues(nVals, _t) + self.JumpPart.generateValues(nVals, _t)

    def generatePaths(self, nPaths=1):
        return self.x0 + ((self.BrownianPart.generatePaths(nPaths) + self.JumpPart.generatePaths(nPaths)).T + self.timePoints * self.mu).T


class NIGProcess(StochasticProcess):

    def __init__(self, T=1, x0=0, nSteps=10000, theta=0., sigma=1., kappa=1.):
        super().__init__(name='NIG Process', T=T, x0=x0, nSteps=nSteps)
        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = x0 or self.x0
        lam = _t ** 2 / self.kappa
        invGauss = sp.invgauss.rvs(mu=_t/lam, scale=lam, size=nVals)  # normally: mu=_t but scipy implementation of IG is different
        brownian = sp.norm.rvs(scale=self.sigma, size=nVals)
        return _x0 + self.theta * invGauss + np.sqrt(invGauss) * brownian

    def generatePaths(self, nPaths=1):
        dt = self.T / (self.nSteps - 1)
        lam = dt ** 2 / self.kappa
        invGauss = sp.invgauss.rvs(mu=dt/lam, scale=lam, size=(nPaths, self.nSteps - 1))
        brownian = sp.norm.rvs(scale=self.sigma, size=(nPaths, self.nSteps - 1))
        increments = np.append(np.ones((nPaths,1)) * self.x0, self.theta * invGauss + np.sqrt(invGauss) * brownian, axis=1)
        return pd.DataFrame(np.cumsum(increments,axis=1).T, index=self.timePoints)









        

    