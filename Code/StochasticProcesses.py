import numpy as np
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

    def generateValues(self, t):
        raise NotImplementedError('Must override generateValues')

    def plotPaths(self, figSize, nPaths):
        plt.figure(figsize=figSize)
        paths = self.generatePaths(nPaths)
        plt.plot(self.timePoints, paths.T)
        plt.ylabel('$S_t$')
        plt.title('{} sample path{} of a {} '.format(nPaths, '' if nPaths == 1 else 's', self.name))




        

    