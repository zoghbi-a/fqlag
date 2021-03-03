import numpy as np
from .base import FqLagBin
from .psdf import Psdf, identify_model

class Psif(FqLagBin):
    """THIS HAS NOT BEEN TESTED"""
    def __init__(self, tarr, yarr, yerr, fql, p1, model=['pl', 'c', 'c'], dt=None, NFQ=8):
        self.NFQ = NFQ
        fqL      = np.logspace(np.log10(fql[0]), np.log10(fql[1]), NFQ)
        self.fq  = (fqL[1:] + fqL[:-1]) / 2.
        self.fqL = np.array(fqL)
        
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        
        super(Psif, self).__init__(t, y, ye, fqL, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5
        
        
        self.psi_func, self.psi_derv, self.psi_npar = identify_model(model[1])
        self.phi_func, self.phi_derv, self.phi_npar = identify_model(model[2])
        self.npar   = self.psi_npar + self.phi_npar
        
        
        pm  = Psdf(tarr[0], yarr[0], yerr[0], fql, model[0], dt, NFQ)
        self.psd     = pm.psd_func(self.fq, p1) # unnormalized
        self.res_1   = pm.covariance(p1)
        self.d_res_1 = np.zeros((self.npar, self.n1, self.n1), np.double)


    def covariance(self, pars):
        psi_p, phi_p = pars[:self.psi_npar], pars[self.psi_npar:]
        psd  = self.psd
        psi  = self.psi_func(self.fq, psi_p)
        phi  = self.phi_func(self.fq, phi_p)
        
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
        psi_p, phi_p = pars[:self.psi_npar], pars[self.psi_npar:]
        
        psd  = self.psd
        
        # nfq and (psi_npar, nfq)
        psi  = self.psi_func(self.fq, psi_p)
        psiD = self.psi_derv(self.fq, psi_p)
        
        # nfq and (phi_npar, nfq)
        phi  = self.phi_func(self.fq, phi_p)
        phiD = self.phi_derv(self.fq, phi_p)
        
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        
        
        n1   = self.n1
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
        
        # t1 #
        # (npar, n1, n1)
        res_1 = self.d_res_1
        
        # t2 #
        res_2 = np.sum(I2_s * np.concatenate([psd * self.norm2 * 2 * psiD * psi, 
                                              phiD*0])[:,None,None,:], -1)
        
        
        # tx #
        # res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        d_cxd = np.concatenate([psd * psiD * self.norm, phiD*0])
        d_phi = np.concatenate([psiD*0, phiD*0 + 1])
                    
        res_x = np.sum(((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[:,None,None,:] + 
                        cxd * d_phi[:,None,None,:] * (-I_s * np.sin(phi) - I_c * np.cos(phi))), -1)
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:], res_x[i,:,:].T]), 
                                  np.vstack([res_x[i,:,:], res_2[i,:,:]])])
                      for i in range(self.npar)]) 
        
        return res
    


class PPsif(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, model=['pl', 'c', 'c'], dt=None, NFQ=8):
        self.NFQ = NFQ
        fqL      = np.logspace(np.log10(fql[0]), np.log10(fql[1]), NFQ)
        self.fq  = (fqL[1:] + fqL[:-1]) / 2.
        self.fqL = np.array(fqL)
        
        t  = np.concatenate(tarr)
        y  = np.concatenate([x-x.mean() for x in yarr])
        ye = np.concatenate(yerr)
        self.n1 = len(tarr[0])
        
        super(PPsif, self).__init__(t, y, ye, fqL, dt)
        self.norm1 = np.mean(yarr[0])**2
        self.norm2 = np.mean(yarr[1])**2
        self.norm  = (self.norm1*self.norm2)**0.5
        
        
        self.psd_func, self.psd_derv, self.psd_npar = identify_model(model[0])
        self.psi_func, self.psi_derv, self.psi_npar = identify_model(model[1])
        self.phi_func, self.phi_derv, self.phi_npar = identify_model(model[2])
        self.npar   = self.psd_npar + self.psi_npar + self.phi_npar
        


    def covariance(self, pars):
        psd_p = pars[:self.psd_npar]
        psi_p = pars[self.psd_npar:(self.psd_npar+self.psi_npar)]
        phi_p = pars[-self.phi_npar:]
        
        psd  = self.psd_func(self.fq, psd_p)
        psi  = self.psi_func(self.fq, psi_p)
        phi  = self.phi_func(self.fq, phi_p)
        
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
        
        psd_p = pars[:self.psd_npar]
        psi_p = pars[self.psd_npar:(self.psd_npar+self.psi_npar)]
        phi_p = pars[-self.phi_npar:]
        
        # nfq and (psd_npar, nfq)
        psd  = self.psd_func(self.fq, psd_p)
        psdD = self.psd_derv(self.fq, psd_p)
        
        # nfq and (psi_npar, nfq)
        psi  = self.psi_func(self.fq, psi_p)
        psiD = self.psi_derv(self.fq, psi_p)
        
        # nfq and (phi_npar, nfq)
        phi  = self.phi_func(self.fq, phi_p)
        phiD = self.phi_derv(self.fq, phi_p)
        
        psd1 = psd * self.norm1
        psd2 = psd * psi**2 * self.norm2
        cxd  = psd * psi * self.norm
        
        
        n1   = self.n1
        I1_s = self.I_s[:n1, :n1]
        I2_s = self.I_s[n1:, n1:]
        I_s  = self.I_s[:n1, n1:]
        I_c  = self.I_c[:n1, n1:]
        
        
        # t1 #
        # (npar, n1, n1)
        res_1 = np.sum(I1_s * np.concatenate([self.norm1 * psdD, 
                                              psiD*0, 
                                              phiD*0])[:,None,None,:], -1)
        
        # t2 #
        res_2 = np.sum(I2_s * np.concatenate([self.norm2 * psdD * psi**2, 
                                              self.norm2 * psd * 2 * psiD * psi, 
                                              phiD*0])[:,None,None,:], -1)
        
        
        # tx #
        # res_x = np.sum(cxd * (I_s * np.cos(phi) - I_c * np.sin(phi)), -1)
        d_cxd = np.concatenate([self.norm * psdD * psi, 
                                self.norm * psd * psiD, 
                                phiD*0])
        d_phi = np.concatenate([psdD*0, psiD*0, phiD*0 + 1])
                    
        res_x = np.sum(((I_s * np.cos(phi) - I_c * np.sin(phi)) * d_cxd[:,None,None,:] + 
                        cxd * d_phi[:,None,None,:] * (-I_s * np.sin(phi) - I_c * np.cos(phi))), -1)
        
        res = np.array([np.hstack([np.vstack([res_1[i,:,:], res_x[i,:,:].T]), 
                                  np.vstack([res_x[i,:,:], res_2[i,:,:]])])
                      for i in range(self.npar)]) 
        
        
        return res