import numpy as np
from .base import FqLagBin
from .psd import Psd, lPsd


class Cxd(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, p1, p2, dt=None, log=False):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
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
        
        
        cxd, phi = np.split(pars, 2)
        if self.islog:
            cxd = np.exp(cxd)
        cxd = cxd * self.norm
      
        n1  = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([self.res_1, res_x.T]), 
                         np.vstack([res_x, self.res_2])])

        return res

    
    def covariance_derivative(self, pars):
        
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


class lCxd(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, p1, p2, dt=None):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(lCxd, self).__init__(t, y, ye, fql, dt)
        
        # constant part of covariance #
        nfq = len(fql) - 1
        pm1 = lPsd(tarr[0], yarr[0], yerr[0], fql, dt)
        self.res_1 = pm1.covariance(p1)
        self.d_res_1 = np.zeros((2, self.n1, self.n1, nfq), np.double)
        
        pm2 = lPsd(tarr[1], yarr[1], yerr[1], fql, dt)
        self.res_2 = pm2.covariance(p2)
        self.d_res_2 = np.zeros((2, self.n-self.n1, self.n-self.n1, nfq), np.double)

        self.norm = pm1.mu * pm2.mu

        
    def covariance(self, pars):
        
        cxd, phi = np.split(pars, 2)
        cxd = np.exp(cxd)
        cxd = cxd * self.norm
      
        n1  = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([self.res_1, res_x.T]), 
                         np.vstack([res_x, self.res_2])])

        return res

    
    def covariance_derivative(self, pars):
                 
        cxd, phi = np.split(pars, 2)
        cxd = np.exp(cxd)
        cxd = cxd * self.norm
        
        nfq = len(phi)
        n1 = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        
        res_cxd = cxd * (I_s * np.cos(phi) - I_c * np.sin(phi))
        res_phi = cxd * (-I_s * np.sin(phi) - I_c * np.cos(phi))
        res_x = np.array([res_cxd, res_phi])
        
        res = np.array([np.hstack([np.vstack([self.d_res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], self.d_res_2[i,:,:,j]])])
                      for i in range(2) for j in range(nfq)]) 
        return res


class Psi(FqLagBin):
    """use psi, fixed psd"""
    
    def __init__(self, tarr, yarr, yerr, fql, p1, dt=None, log=True):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(Psi, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5
        self.islog = log

        nfq = len(fql) - 1
        pm  = Psd(tarr[0], yarr[0], yerr[0], fql, dt, log)
        self.psd   = np.exp(p1) if log else np.array(p1)
        self.res_1 = pm.covariance(p1)
        self.d_res_1 = np.zeros((2, self.n1, self.n1, nfq), np.double)

        
    def covariance(self, pars):
         
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
        
        
        res_1 = self.res_1
        res_2 = np.sum(psd2 * I2_s, -1)
        
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
        
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
    

class lPsi(FqLagBin):
    """use log psi, fixed psd"""
    
    def __init__(self, tarr, yarr, yerr, fql, p1, dt=None):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(lPsi, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5

        nfq = len(fql) - 1
        pm  = lPsd(tarr[0], yarr[0], yerr[0], fql, dt)
        self.psd   = np.exp(p1)
        self.res_1 = pm.covariance(p1)
        self.d_res_1 = np.zeros((2, self.n1, self.n1, nfq), np.double)

        
    def covariance(self, pars):
         
        psi, phi = np.split(pars, 2)
        psd  = self.psd
        psi  = np.exp(psi)
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        
        n1   = self.n1
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
        
        res_1 = self.res_1
        res_2 = np.sum(psd2 * I2_s, -1)
        
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
        
        psi, phi = np.split(pars, 2)
        psd  = self.psd
        psi  = np.exp(psi)
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        nfq  = len(phi)
        
        n1   = self.n1
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
    
        # t1 #
        # (2, n1, n1, nfq)
        res_1 = self.d_res_1
        
        # t2 #
        res_2 = np.array([I2_s*2*psd2, I2_s*0])
        
        
        # tx #
        d_cxd = np.array([cxd, phi*0])
        d_phi = np.array([phi*0, phi*0+1])
    
        res_x = ((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[:,None,None,:] + 
                  cxd * d_phi[:,None,None,:] * (-I_s * np.sin(phi) - I_c * np.cos(phi)))
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], res_2[i,:,:,j]])])
                      for i in range(2) for j in range(nfq)]) 
        return res


class CxdRI(FqLagBin):
    """Use real and imaginary parts"""
    
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
        


