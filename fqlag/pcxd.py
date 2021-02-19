import numpy as np
from .base import FqLagBin
from .psd import Psd


class PCxd(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None, log=False):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(PCxd, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5
        self.islog = log

        
    def covariance(self, pars):
        
        psd1, psd2, cxd, phi = np.split(pars, 4)
        psd1 = self.norm1 * (np.exp(psd1) if self.islog else np.array(psd1))
        psd2 = self.norm2 * (np.exp(psd2) if self.islog else np.array(psd2))
        cxd  = self.norm  * (np.exp(cxd)  if self.islog else np.array(cxd))
      
        n1  = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]

        res_1 = np.sum(psd1 * self.I_s[:n1, :n1], -1)
        res_2 = np.sum(psd2 * self.I_s[n1:, n1:], -1)
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
                 
        psd1, psd2, cxd, phi = np.split(pars, 4)
        psd1 = self.norm1 * (np.exp(psd1) if self.islog else np.array(psd1))
        psd2 = self.norm2 * (np.exp(psd2) if self.islog else np.array(psd2))
        cxd  = self.norm  * (np.exp(cxd)  if self.islog else np.array(cxd))

        dpsd1 = psd1 if self.islog else self.norm1
        dpsd2 = psd2 if self.islog else self.norm2
        dcxd  = cxd  if self.islog else self.norm
        
        nfq = len(phi)
        n1 = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        res_p1 = dpsd1 *  self.I_s[:n1, :n1]
        res_1  = np.array([res_p1, res_p1*0, res_p1*0, res_p1*0])
        res_p2 = dpsd2 *  self.I_s[n1:, n1:]
        res_2  = np.array([res_p2*0, res_p2, res_p2*0, res_p2*0])


        res_cxd  = dcxd * (I_s * np.cos(phi) - I_c * np.sin(phi))
        res_phi  = cxd * (-I_s * np.sin(phi) - I_c * np.cos(phi))
        res_x = np.array([res_cxd*0, res_cxd*0, res_cxd, res_phi])
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                   np.vstack([res_x[i,:,:,j], res_2[i,:,:,j]])])
                      for i in range(4) for j in range(nfq)]) 
        return res


class lPCxd(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(lPCxd, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5

        
    def covariance(self, pars):
        
        psd1, psd2, cxd, phi = np.split(pars, 4)
        psd1 = np.exp(psd1) * self.norm1
        psd2 = np.exp(psd2) * self.norm2
        cxd  = np.exp(cxd)  * self.norm
      
        n1  = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]

        res_1 = np.sum(psd1 * self.I_s[:n1, :n1], -1)
        res_2 = np.sum(psd2 * self.I_s[n1:, n1:], -1)
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
                 
        psd1, psd2, cxd, phi = np.split(pars, 4)
        psd1 = np.exp(psd1) * self.norm1
        psd2 = np.exp(psd2) * self.norm2
        cxd  = np.exp(cxd)  * self.norm
        
        nfq = len(phi)
        n1 = self.n1
        I_s = self.I_s[:n1, n1:]
        I_c = self.I_c[:n1, n1:]
    
        res_p1 = psd1 *  self.I_s[:n1, :n1]
        res_1  = np.array([res_p1, res_p1*0, res_p1*0, res_p1*0])
        res_p2 = psd2 *  self.I_s[n1:, n1:]
        res_2  = np.array([res_p2*0, res_p2, res_p2*0, res_p2*0])


        res_cxd  = cxd * (I_s * np.cos(phi) - I_c * np.sin(phi))
        res_phi  = cxd * (-I_s * np.sin(phi) - I_c * np.cos(phi))
        res_x = np.array([res_cxd*0, res_cxd*0, res_cxd, res_phi])
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                   np.vstack([res_x[i,:,:,j], res_2[i,:,:,j]])])
                      for i in range(4) for j in range(nfq)]) 
        return res


class PPsi(FqLagBin):
    """use psi"""
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None, log=False):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(PPsi, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5
        self.islog = log

        
    def covariance(self, pars):
         
        psd, psi, phi = np.split(pars, 3)
        psd = np.exp(psd) if self.islog else np.array(psd)
        psi = np.exp(psi) if self.islog else np.array(psi)
        psd1 = self.norm1 * psd
        psd2 = self.norm2 * psd * psi**2
        cxd  = self.norm  * psd * psi
        
        n1   = self.n1
        I1_s = self.I_s[:n1, :n1]
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
        
        res_1 = np.sum(psd1 * I1_s, -1)
        res_2 = np.sum(psd2 * I2_s, -1)
        
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
        
        psd, psi, phi = np.split(pars, 3)
        psd = np.exp(psd) if self.islog else np.array(psd)
        psi = np.exp(psi) if self.islog else np.array(psi)
        psd1 = self.norm1 * psd
        psd2 = self.norm2 * psd * psi**2
        cxd  = self.norm  * psd * psi
        nfq  = len(phi)

        dpsd  = psd if self.islog else 1.0
        dpsi  = psi if self.islog else 1.0
        dpsd1 = self.norm1 * dpsd
        dpsd2dpsd = self.norm2 * dpsd * psi**2
        dpsd2dpsi = self.norm2 * 2 * dpsi * psd * psi 
        dcxddpsd  = self.norm * dpsd * psi
        dcxddpsi  = self.norm * dpsi * psd
        
        n1   = self.n1
        I1_s = self.I_s[:n1, :n1]
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
    
        # t1 #
        # (3, n1, n1, nfq)
        res_1 = np.array([I1_s*dpsd1, I1_s*0, I1_s*0]) 
        
        # t2 #
        res_2 = np.array([I2_s*dpsd2dpsd, I2_s*dpsd2dpsi, I2_s*0])
        
        
        # tx #
        d_cxd = np.array([dcxddpsd, dcxddpsi, phi*0])
        d_phi = np.array([phi*0, phi*0, phi*0+1])
    
        res_x = ((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[:,None,None,:] + 
                  cxd * d_phi[:,None,None,:] * (-I_s * np.sin(phi) - I_c * np.cos(phi)))
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], res_2[i,:,:,j]])])
                      for i in range(3) for j in range(nfq)]) 
        return res


class lPPsi(FqLagBin):
    """use psi"""
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None):
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        super(lPPsi, self).__init__(t, y, ye, fql, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5

        
    def covariance(self, pars):
         
        psd, psi, phi = np.split(pars, 3)
        psd  = np.exp(psd)
        psi  = np.exp(psi)
        psd1 = psd * self.norm1
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        
        n1   = self.n1
        I1_s = self.I_s[:n1, :n1]
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
        
        res_1 = np.sum(psd1 * I1_s, -1)
        res_2 = np.sum(psd2 * I2_s, -1)
        
    
        res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        res = np.hstack([np.vstack([res_1, res_x.T]), 
                         np.vstack([res_x, res_2])])

        return res

    
    def covariance_derivative(self, pars):
        
        psd, psi, phi = np.split(pars, 3)
        psd  = np.exp(psd)
        psi  = np.exp(psi)
        psd1 = psd * self.norm1
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        nfq  = len(phi)
        
        n1   = self.n1
        I1_s = self.I_s[:n1, :n1]
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
    
        # t1 #
        # (3, n1, n1, nfq)
        res_1 = np.array([psd1*I1_s, I1_s*0, I1_s*0])
        
        # t2 #
        res_2 = np.array([psd2*I2_s, I2_s*2*psd2, I2_s*0])
        
        
        # tx #
        d_cxd = np.array([cxd, cxd, phi*0])
        d_phi = np.array([phi*0, phi*0, phi*0+1])
    
        res_x = ((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[:,None,None,:] + 
                  cxd * d_phi[:,None,None,:] * (-I_s * np.sin(phi) - I_c * np.cos(phi)))
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:,j], res_x[i,:,:,j].T]), 
                                  np.vstack([res_x[i,:,:,j], res_2[i,:,:,j]])])
                      for i in range(3) for j in range(nfq)]) 
        return res

