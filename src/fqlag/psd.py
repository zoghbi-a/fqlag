"""Class for piece-wise frequency-dependent power spectra"""

import numpy as np

from .base import FqLagBin


class Psd(FqLagBin):
    """Model for calculating Power Spectral Density (PSD) 
        as peicewise values at some frequency bins
    """

    def __init__(self,
                 tarr: np.ndarray,
                 yarr: np.ndarray,
                 yerr: np.ndarray,
                 fql: np.ndarray,
                 dt: float = None,
                 log: bool = False):
        """Initialize a Psd instance to calculate the power spectrum.

        This class defines the covariance kernel, and the calculations are done
        in FqLagBin that Psd inherits from.

        Parameters
        ----------
        tarr: np.ndarray
            a numpy array giving the time axis of the light curve.
        yarr: np.ndarray
            a numpy array giving the count rate or flux
        yerr: np.ndarray
            a numpy array giving the 1-sigma measurement uncertainity
            in the count rate or flux.
        fql: np.ndarray
            a numpy array of frequency bin boundaries
        dt: float
            sampling time of the light curves. If given, corrections to sampling
            bias is applied, otherwise, we don't apply it.
        log: bool
            if True, the model parameters are the log of the psd values,
            otherwise, model the psd values.

        """

        # initialize the parent class #
        super().__init__(tarr, yarr, yerr, fql, dt)
        # set the norm; this ensures the psd values returned are normalized
        # according to the rms normalization (e.g. Vaughan et al. 2003)
        self.norm = self.mean**2
        self.islog = log
        self.params = {'fql':fql, 'dt':dt, 'log':log}


    def covariance(self, pars: np.ndarray):
        """Covariance kernel function for modeling the power spectrum.
        
        The model parameters are the psd values with rms normalization at 
        the geometric center of the frequency bins. 
        If islog: the parameters are the logs of the psd values
        There are npar (= self.nfq) parameters

        cov = sum_i(P_i * I_s[:,:,i]), with i spanning the frequency bins
        and I_s is the Fourier integral defined in FqLagBin
        
        
        Parameters
        ----------
        pars: np.ndarray
            parameters that define the covariance kernel.

        Returns
        -------
        A covariance matrix of shape: (self.npoints, self.npoints)

        """
        # get normalized psd values #
        psd  = np.exp(pars) if self.islog else np.array(pars)
        psd *= self.norm

        res = np.sum(psd * self.I_s, -1)
        return res


    def covariance_derivative(self, pars: np.ndarray):
        psd  = np.exp(pars) if self.islog else np.array(pars)*0 + 1
        return (psd * self.I_s).T * self.norm
