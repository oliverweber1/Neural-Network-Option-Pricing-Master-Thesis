from StochasticProcesses import *

class MultivariateProcess(StochasticProcess):

    #def __init__(self, name, x0, d, T=1, nSteps=10000):
    #    super().__init__(name, x0, T, nSteps, d)

    def generatePath(self):
        raise NotImplementedError('Must override generatePaths')

    def plotPath(self, figSize=(10,5)):
        plt.figure(figsize=figSize)
        paths = self.generatePath()
        plt.plot(paths)
        plt.ylabel('$S_t$')
        plt.title('Sample paths of a {}-dimensional {} '.format(self.d, self.name))


class CorrelatedBrownianMotion(MultivariateProcess):

    def __init__(self, d, x0=0., T=1, mu=None, sigma=None, Corr=None, nSteps=10000):
        super().__init__('Correlated Brownian Motion', x0, T, nSteps, d)
        #self.Factors = np.eye(d) if Corr is None else np.linalg.cholesky(Corr)
        self.mu = np.zeros((d, 1)) if mu is None else np.reshape(mu, (d, 1))
        self.sigma = np.ones((d, 1)) if sigma is None else np.reshape(sigma, (d, 1))
        _Corr = np.eye(d) if Corr is None else Corr
        self.cov = np.diag(self.sigma.reshape(d)) @ _Corr @ np.diag(self.sigma.reshape(d))

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = self.x0 if x0 is None else x0.reshape((self.d,1))
        #normalVals = self.Factors @ sp.norm.rvs(scale=np.sqrt(_t), size=(nVals, self.d, 1))
        #return _x0 + self.mu * _t + self.sigma * normalVals
        normalVals = sp.multivariate_normal.rvs(mean=np.zeros(self.d), cov=self.cov, size=nVals).reshape((nVals, self.d, 1))
        return _x0 + self.mu * _t + np.sqrt(_t) * normalVals
        

    def generatePath(self):
        dt = self.T / (self.nSteps - 1)
        #increments = np.append(np.zeros((self.d, 1)), sp.norm.rvs(scale=np.sqrt(dt), size=(self.d,self.nSteps - 1)), axis=1)
        #paths = self.x0 + self.mu * self.timePoints + self.sigma * (self.Factors @ np.cumsum(increments, axis=1))
        increments = np.append(np.zeros((self.d, 1)), sp.multivariate_normal.rvs(mean=np.zeros(self.d), cov=self.cov, size=self.nSteps-1).T, axis=1)
        paths = self.x0 + self.mu * self.timePoints + np.sqrt(dt) * np.cumsum(increments, axis=1)
        return pd.DataFrame(paths.T, index=self.timePoints)


class MultivariateCompoundPoissonProcess(MultivariateProcess):
    """
    Only for normally distributed jumps
    """
    def __init__(self, d, lam, x0=0., T=1, mu=None, sigma=None, Corr=None, nSteps=10000):
        super().__init__('Multivariate Compound Poisson Process', x0, T, nSteps, d)
        self.lam = lam
        _mu = np.zeros(d) if mu is None else np.reshape(mu, d)
        _sigma = np.eye(d) if sigma is None else np.diag(np.reshape(sigma, d))
        _Corr = np.eye(d) if Corr is None else Corr
        cov = _sigma @ _Corr @ _sigma
        self.jumpSizeRV = sp.multivariate_normal(mean=_mu, cov=cov)

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = self.x0 if x0 is None else x0.reshape((self.d, 1))
        numOfJumps = sp.poisson.rvs(mu=self.lam * _t, size=nVals)
        maxJumps = np.max(numOfJumps)
        jumpSizes = self.jumpSizeRV.rvs(size=(nVals, maxJumps))
        if nVals == 1:
            jumpSizes = np.reshape(jumpSizes, (1, maxJumps, self.d))
        jumpSizes[numOfJumps[:,None] <= np.arange(jumpSizes.shape[1])] = 0
        return _x0 + jumpSizes.sum(axis=1).reshape((nVals, self.d, 1))

    def generatePath(self):
        paths = pd.DataFrame(np.ones((len(self.timePoints), self.d)) * self.x0.T, index=self.timePoints)
        numberOfJumps = sp.poisson.rvs(mu=self.lam * self.T)
        jumpTimes = sp.uniform.rvs(scale=self.T, size=numberOfJumps)
        for jump in jumpTimes:
            paths[paths.index > jump] += self.jumpSizeRV.rvs()
        return paths


class MultivariateJumpDiffusionProcess(MultivariateProcess):

    def __init__(self, BrownianPart, JumpPart):
        assert (BrownianPart.timePoints == JumpPart.timePoints).all(), 'Time grids of Brownian part and jump part do not coincide!'
        super().__init__('Multivariate Jump Diffusion Process', 0., BrownianPart.T, BrownianPart.nSteps, BrownianPart.d)
        self.BrownianPart = BrownianPart
        self.JumpPart = JumpPart

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        return self.BrownianPart.generateValues(nVals, _t) + self.JumpPart.generateValues(nVals, _t)

    def generatePath(self):
        return self.BrownianPart.generatePath() + self.JumpPart.generatePath()
    
class MultivariateNIGProcess(MultivariateProcess):

    def __init__(self, d, x0=0., T=1, nSteps=10000, kappa=0., theta=None, sigma=None, Corr=None):
        super().__init__('Multivariate Normal Inverse Gaussian Process', x0, T, nSteps, d)
        self.theta = np.zeros(d) if theta is None else np.reshape(theta, d)
        self.sigma = np.ones(d) if sigma is None else np.reshape(sigma, d)
        _Corr = np.eye(d) if Corr is None else Corr
        self.cov = np.diag(self.sigma) @ _Corr @ np.diag(self.sigma)
        self.kappa = kappa

    def generateValues(self, nVals=1, t=None, x0=None):
        _t = t or self.T
        _x0 = self.x0 if x0 is None else x0
        lam = _t ** 2 / self.kappa
        invGauss = np.tile(sp.invgauss.rvs(mu=_t/lam, scale=lam, size=nVals).reshape((nVals, 1)), self.d)  # normally: mu=_t but scipy implementation of IG is different
        brownian = sp.multivariate_normal.rvs(mean=np.zeros(self.d), cov=self.cov, size=nVals).reshape((nVals, self.d))
        return (_x0.reshape(self.d) + self.theta * invGauss + np.sqrt(invGauss) * brownian).reshape(nVals, self.d, 1)

    def generatePath(self):
        dt = self.T / (self.nSteps - 1)
        lam = dt ** 2 / self.kappa
        invGauss = sp.invgauss.rvs(mu=dt/lam, scale=lam, size=self.nSteps-1)
        brownian = sp.multivariate_normal.rvs(mean=np.zeros(self.d), cov=self.cov, size=self.nSteps-1).T
        increments = np.append(self.x0, self.theta.reshape((self.d, 1)) * invGauss + np.sqrt(invGauss) * brownian, axis=1)
        return pd.DataFrame(np.cumsum(increments, axis=1).T, index=self.timePoints)