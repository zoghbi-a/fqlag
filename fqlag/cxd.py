import numpy as np
from .base import FqLagBin
from .psd import Psd


class Cxd(FqLagBin):
    """Model for calculating Cross Spectral Density (CXD) and phase lag
        of two light curves as peicewise values at some given frequency bins

    This class defines the covariance kernel, and the calculations are done
    in FqLagBin that Cxd inherits from.
    An alternative modeling can be done by calculating the amplitude and phase
    of the transfer function

    Args:
        tarr: a list of two numpy arrays giving the time axis of each light curve.
        yarr: a list of two numpy arrays giving the count rate or flux of each light curve
        yerr: a list of two numpy arrays giving the 1-sigma measurement uncertainity
            in the count rate or flux of each light curve.
        fql: a numpy array of frequency bin boundaries
        p1, p2: the PSD parameter values corresponding to the frequency bins defined 
            in fql for each light curve. These should be calculated beforehand using
            Psd class. dt and log parameters here should be consistent with those used
            to obtain p1 and p2
        dt: sampling time of the light curves. If given, corrections to sampling
            bias is applied, otherwise, we don't apply it.
        log: if True, the cross spectrum parameters are in log units (p1 and p2
            are assumed to be log too in this case), otherwise,  the cxd parameters
            are in linear scale. The phase parameters have linear scale.

    """
    
    def __init__(self, tarr, yarr, yerr, fql, p1, p2, dt=None, log=False):
        # concatenate the arrays from the two light curves
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        # initialize the parent class #
        super(Cxd, self).__init__(t, y, ye, fql, dt)
        
        # constant part of covariance #
        nfq = len(fql) - 1
        pm1 = Psd(tarr[0], yarr[0], yerr[0], fql, dt, log=log)
        self.res_1 = pm1.covariance(p1)
        self.d_res_1 = np.zeros((2, self.n1, self.n1, nfq), np.double)
        
        pm2 = Psd(tarr[1], yarr[1], yerr[1], fql, dt, log=log)
        self.res_2 = pm2.covariance(p2)
        self.d_res_2 = np.zeros((2, self.n-self.n1, self.n-self.n1, nfq), np.double)

        self.norm = pm1.mu * pm2.mu
        self.islog = log

        
    def covariance(self, pars):
        """Covariance kernel function for modeling the cross spectrum and phase lag.
        
        The model parameters are the cross spectrum values with rms normalization 
        (first half of the parameters) and the phase delay (second half) at the 
        geometric center of the frequency bins. 
        If islog: the cross spectrum parameters are in log units
        There are npar (= 2*self.nfq) parameters

        The covariance has 4 blocks:
        [ [p1Cov] [Cov^T]
          [Cov]   [p2Cov] ]
        where p1Cov is the covariance resulting from the psd of the first light curve,
        which is constant here and defined by the psd parameter input in p1.
        p2Cov is the corresponding covariance from the psd of the second light curve,
        corresponding to p2.
        Cov is the crovariance matrix resulting from the cross spectrum and phase delay.
        Cov = sum_i(C_i * (I_s[:,:,i]*cos(Phi_i) - I_c[:,:,i]*cos(Phi_i)) ), 
        with i spanning the frequency bins
        and I_s, I_c are the Fourier integrals defined in FqLagBin
        
        
        Args:
            pars: parameters that defined the covariance kernel.
                pars[:nfq] are the cross spectrum values at nfq bins
                pars[nfq:] are the phase delays values at nfq bins

        Returns:
            A covariance matrix of shape: (self.n, self.n)
            where self.n is the sum of the lengths of the two light curves

        """
        
        # get normalized cxd values #
        cxd, phi = np.split(pars, 2)
        if self.islog:
            cxd = np.exp(cxd)
        cxd = cxd * self.norm
      
        # fill the constant parts of the covariance matrix
        n1  = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        # the covariance block from the cross spectra and phase delays
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([self.res_1, res_x.T]), 
                         np.vstack([res_x, self.res_2])])

        return res

    
    def covariance_derivative(self, pars):
        """First Derivative of the covariance kernel function 
        for modeling the cross spectrum and phase delays.

        This calculates the derivative of covariance values with respect
        to each model parameter
        

        Args:
            pars: parameters of the covariance kernel.
                pars[:nfq] are the cross spectrum values at nfq bins
                pars[nfq:] are the phase delays values at nfq bins

        Returns:
            a matrix of first derivatives with shape (npar, self.n, self.n)
            where self.n is the sum of the lengths of the two light curves

        """
        cxd, phi = np.split(pars, 2)
        if self.islog:
            cxd = np.exp(cxd)
        cxd  = cxd * self.norm
        dcxd = cxd if self.islog else self.norm
        nfq = len(phi)
        
        n1 = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        
        res_cxd = dcxd * ( I_s * np.cos(phi) - I_c * np.sin(phi))
        res_phi = cxd  * (-I_s * np.sin(phi) - I_c * np.cos(phi))
        res_x = np.array([res_cxd, res_phi])
        
        res = np.array([np.hstack([np.vstack([self.d_res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], self.d_res_2[i,:,:,j]])])
                      for i in range(2) for j in range(nfq)]) 
        return res



class Psi(FqLagBin):
    """Model for calculating Amplitude (Psi) and phase lag
        of two light curves as peicewise values at some given frequency bins

    This class defines the covariance kernel, and the calculations are done
    in FqLagBin that Psi inherits from.
    An alternative modeling can be done by calculating the cross spectrum and phase
    using Cxd

    Args:
        tarr: a list of two numpy arrays giving the time axis of each light curve.
        yarr: a list of two numpy arrays giving the count rate or flux of each light curve
        yerr: a list of two numpy arrays giving the 1-sigma measurement uncertainity
            in the count rate or flux of each light curve.
        fql: a numpy array of frequency bin boundaries
        p1: the PSD parameter values corresponding to the frequency bins defined 
            in fql for the first light curve, typically the reference. 
            These should be calculated beforehand using Psd class. 
        dt: sampling time of the light curves. If given, corrections to sampling
            bias is applied, otherwise, we don't apply it.
        log: if True, the amplitude parameters are in log units (p1 is assumed 
            to be log too), otherwise,  the psi parameters
            are in linear scale. The phase parameters have linear scale."""
    
    def __init__(self, tarr, yarr, yerr, fql, p1, dt=None, log=True):
        # concatenate the arrays from the two light curves
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(Psi, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5
        self.islog = log

        # constant part of covariance #
        nfq = len(fql) - 1
        pm  = Psd(tarr[0], yarr[0], yerr[0], fql, dt, log)
        self.psd   = np.exp(p1) if log else np.array(p1)
        self.res_1 = pm.covariance(p1)
        self.d_res_1 = np.zeros((2, self.n1, self.n1, nfq), np.double)

        
    def covariance(self, pars):
        """Covariance kernel function for modeling the amplitude and phase of 
        the transfer function.
        
        The model parameters are the amplitude (first half of the parameters) 
        and the phase delay (second half) at the geometric center of the 
        frequency bins. 
        If islog: the amplitude parameters are in log units
        There are npar (= 2*self.nfq) parameters

        The covariance has 4 blocks:
        [ [p1Cov] [Cov^T]
          [Cov]   [p2Cov] ]
        where p1Cov is the covariance resulting from the psd of the first light curve,
        which is constant here and defined by the psd parameter input in p1.
        p2Cov is the covariance from the psd of the second light curve,
        corresponding to parameters p1*psi**2 (with a normalizing factor).
        Cov is the crovariance matrix resulting from the amplitude and phase delay.
        Cov = sum_i(p1*psi * (I_s[:,:,i]*cos(Phi_i) - I_c[:,:,i]*cos(Phi_i)) ), 
        with i spanning the frequency bins
        and I_s, I_c are the Fourier integrals defined in FqLagBin
        
        
        Args:
            pars: parameters that defined the covariance kernel.
                pars[:nfq] are the (positive) amplitude values of the transfer function
                     at nfq bins
                pars[nfq:] are the phase delays values at nfq bins

        Returns:
            A covariance matrix of shape: (self.n, self.n)
            where self.n is the sum of the lengths of the two light curves

        """
        psi, phi = np.split(pars, 2)
        if self.islog:
            psi = np.exp(psi)
        psd  = self.psd
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        

        n1   = self.n1
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
        # fill the psd parts of the covariance matrix
        res_1 = self.res_1
        res_2 = np.sum(psd2 * I2_s, -1)
        
        # then fill in the cross-covariance values
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
        """First Derivative of the covariance kernel function 
        for modeling the amplitude and phase of the transfer function between
        two light curves.

        This calculates the derivative of covariance values with respect
        to each model parameter
        

        Args:
            pars: parameters of the covariance kernel.
                pars[:nfq] are the cross spectrum values at nfq bins
                pars[nfq:] are the phase delays values at nfq bins

        Returns:
            a matrix of first derivatives with shape (npar, self.n, self.n)
            where self.n is the sum of the lengths of the two light curves

        """
        psi, phi = np.split(pars, 2)
        if self.islog:
            psi = np.exp(psi)
        psd  = self.psd
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        nfq  = len(phi)

        dpsi  = psi if self.islog else 1.0
        dpsd2 = 2 * dpsi * psd * psi * self.norm2
        dcxd  = dpsi * psd * self.norm
        
        n1   = self.n1
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
    
        # t1 #
        # (2, n1, n1, nfq)
        res_1 = self.d_res_1
        
        # t2 #
        res_2 = np.array([I2_s*dpsd2, I2_s*0])
        
        
        # tx #
        d_cxd = np.array([dcxd, phi*0])
        d_phi = np.array([phi*0, phi*0+1])
    
        res_x = ((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[:,None,None,:] + 
                  cxd * d_phi[:,None,None,:] * (-I_s * np.sin(phi) - I_c * np.cos(phi)))
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], res_2[i,:,:,j]])])
                      for i in range(2) for j in range(nfq)]) 
        return res
    


class CxdRI(FqLagBin):
    """Use real and imaginary parts; THIS HAS NOT BEEN TESTED"""
    
    def __init__(self, tarr, yarr, yerr, fql, p1, p2, dt=None):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(CxdRI, self).__init__(t, y, ye, fql, dt)
        
        # constant part of covariance #
        nfq = len(fql) - 1
        pm1 = Psd(tarr[0], yarr[0], yerr[0], fql, dt)
        self.res_1 = pm1.covariance(p1)
        self.d_res_1 = np.zeros((2, self.n1, self.n1, nfq), np.double)
        
        pm2 = Psd(tarr[1], yarr[1], yerr[1], fql, dt)
        self.res_2 = pm2.covariance(p2)
        self.d_res_2 = np.zeros((2, self.n-self.n1, self.n-self.n1, nfq), np.double)

        self.norm = pm1.mu * pm2.mu

        
    def covariance(self, pars):
        
        
        re, im = np.split(pars, 2)
        re,im = re*self.norm, im*self.norm
        cxd = (re**2 + im**2)**0.5
        phi = np.arctan2(im, re)
      
        n1  = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([self.res_1, res_x.T]), 
                         np.vstack([res_x, self.res_2])])

        return res

    
    def covariance_derivative(self, pars):
        
        re, im = np.split(pars, 2)
        re,im = re*self.norm, im*self.norm
        cxd = (re**2 + im**2)**0.5
        phi = np.arctan2(im, re)
        nfq = len(phi)
        
        n1 = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        d_cxd = np.array([re, im]) / cxd
        d_phi = np.array([-im, re]) / cxd**2
    
        # res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res_re = ((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[0] + 
                  cxd * d_phi[0] * (-I_s * np.sin(phi) - I_c * np.cos(phi)))
        res_im = ((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[1] + 
                  cxd * d_phi[1] * (-I_s * np.sin(phi) - I_c * np.cos(phi)))
          
        res_x = np.array([res_re, res_im])*self.norm
        
        res = np.array([np.hstack([np.vstack([self.d_res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], self.d_res_2[i,:,:,j]])])
                      for i in range(2) for j in range(nfq)]) 
        return res
        


