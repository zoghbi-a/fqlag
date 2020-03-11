import numpy as np
from .base import FqLagBin


class Psd(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None):
        super(Psd, self).__init__(tarr, yarr, yerr, fql, dt)
        self.norm = self.mu**2


    def covariance(self, pars):
        psd  = np.array(pars)
        psd *= self.norm 

        res = np.sum(psd * self.I_s, -1)
        return res

    def covariance_derivative(self, pars):
        return self.I_s.T * self.norm


class lPsd(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None):
        super(lPsd, self).__init__(tarr, yarr, yerr, fql, dt)
        self.norm = self.mu**2
    
    def covariance(self, pars):
        psd  = np.exp(pars)
        psd *= self.norm
                
        res = np.sum(psd * self.I_s, -1)
        return res

    def covariance_derivative(self, pars):
        psd = np.exp(pars)
        psd *= self.norm
        
        return (psd * self.I_s).T

