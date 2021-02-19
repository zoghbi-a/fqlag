import numpy as np
from .base import FqLagBin


class Psd(FqLagBin):
    """Model for calculating Power Spectral Density (PSD) 
        as peicewise values at some frequency bins

    This class defines the covariance kernel, and the calculations are done
    in FqLagBin that Psd inherits from.

    Args:
        tarr: a numpy array giving the time axis of the light curve.
        yarr: a numpy array giving the count rate or flux
        yerr: a numpy array giving the 1-sigma measurement uncertainity
            in the count rate or flux.
        fql: a numpy array of frequency bin boundaries
        dt: sampling time of the light curves. If given, corrections to sampling
            bias is applied, otherwise, we don't apply it.
        log: if True, the model parameters are the log of the psd values,
            otherwise, model the psd values.

    """
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None, log=False):
        # initialize the parent class #
        super(Psd, self).__init__(tarr, yarr, yerr, fql, dt)
        # set the norm; this ensures the psd values returned are normalized
        # according to the rms normalization (e.g. Vaughan et al. 2003)
        self.norm = self.mu**2
        self.islog = log


    def covariance(self, pars):
        """Covariance kernel function for modeling the power spectrum.
        
        The model parameters are the psd values with rms normalization at 
        the geometric center of the frequency bins. 
        If islog: the parameters are the logs of the psd values
        There are npar (= self.nfq) parameters

        cov = sum_i(P_i * I_s[:,:,i]), with i spanning the frequency bins
        and I_s is the Fourier integral defined in FqLagBin
        
        
        Args:
            pars: parameters that defined the covariance kernel.

        Returns:
            A covariance matrix of shape: (self.n, self.n)

        """
        # get normalized psd values #
        psd  = np.exp(pars) if self.islog else np.array(pars)
        psd *= self.norm 

        res = np.sum(psd * self.I_s, -1)
        return res


    def covariance_derivative(self, pars):
        """First Derivative of the covariance kernel function 
        for modeling the power spectrum.

        This calculates the derivative of covariance values with respect
        to each model parameter
        

        Args:
            pars: parameters of the covariance kernel.

        Returns:
            a matrix of first derivatives with shape (npar, self.n, self.n)

        """
        psd  = np.exp(pars) if self.islog else np.array(pars)*0 + 1
        return (psd * self.I_s).T * self.norm


