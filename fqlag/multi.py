import numpy as np

class multiFqLagBin:
    """A class for fitting multiple light curves simultaneously"""
    
    def __init__(self, models):
        self.nmod = len(models)
        self.mods = [m for m in models]
        
    def loglikelihood(self, pars):
        return np.sum([m.loglikelihood(pars) for m in self.mods])

    def loglikelihood_derivative(self, pars, calc_fisher=True):
        lgh = [m.loglikelihood_derivative(pars, calc_fisher) for m in self.mods]

        logLike = np.sum([l[0] for l in lgh])
        grad    = np.sum([l[1] for l in lgh], axis=0)

        if not calc_fisher:
            return logLike, grad

        fisher  = np.sum([l[2] for l in lgh], axis=0)
        return logLike, grad, fisher