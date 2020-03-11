
import numpy as np
import scipy.linalg as alg
import scipy.special as sp


class FqLagBase:
    """A base class that does the likelihood calculations"""
    
    def __init__(self, tarr, yarr, yerr):
        self.n    = tarr.shape[0]
        self.tarr = tarr
        self.mu   = yarr.mean()
        self.yarr = yarr - self.mu
        self.sig2 = yerr**2
        self.tmat = tarr - np.expand_dims(tarr, 1)
        self.yyT  = self.yarr * np.expand_dims(self.yarr, 1)
        self.likelihood_vars = None
        self.delta = 0.0
        
    
    def covariance(self, pars):
        """Kernel function. It is meant to be overloaded by child classes"""
        return np.abs(self.tmat) * pars[0] + pars[1]
    
    def covariance_derivative(self, pars):
        return np.array([
            np.abs(self.tmat),
            np.abs(self.tmat)*0 + 1
        ])

    def covariance_derivative2(self, pars):
        return np.array([[self.tmat*0, self.tmat*0],
                         [self.tmat*0, self.tmat*0]])

    
    def loglikelihood(self, pars):
        """Call kernel to calculate covariance, 
        then calculate log-likelihood
        """
        cov = self.covariance(pars)
        cov += np.diag(self.sig2 + self.delta)
        chol = alg.cho_factor(cov, lower=False)
        logDet = 2 * np.sum(np.log(np.diag(chol[0])))
        chi2 = np.dot(self.yarr, alg.cho_solve(chol, self.yarr))
        logLike = -0.5 * ( chi2 + logDet + self.n*np.log(2*np.pi) )
        # simple caching #
        self.likelihood_vars = [np.array(pars), cov, chol, logLike]
        return logLike
    
    def loglikelihood_derivative(self, pars, calc_fisher=True):
        """Calculate the derivate of the loglikelihood"""
        # see Bond+98 #
        if (self.likelihood_vars is None or 
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        p, cov, chol, logLike = self.likelihood_vars
        
        # gradient #
        yyTmC  = self.yyT - cov
        dcov   = self.covariance_derivative(pars)
        icov   = alg.cho_solve(chol, np.identity(self.n))
        cidc   = [alg.blas.dsymm(1.0, icov, dc) for dc in dcov]
        cidcci = [alg.blas.dsymm(1.0, icov, ci, side=1) for ci in cidc]
        grad   = np.array([0.5 * np.trace(np.dot(yyTmC, ci)) for ci in cidcci])
        
        if not calc_fisher:
            return logLike, grad


        # fisher matrix, approximate hessian #
        npar   = len(grad)
        fisher = np.zeros((npar, npar))
        for i in range(npar):
            for j in range(i+1):
                fisher[i,j] = 0.5 * np.trace(alg.blas.dsymm(1.0, cidc[i], cidc[j]))
                fisher[j,i] = fisher[i,j]
        return logLike, grad, fisher


    def loglikelihood_derivative2(self, pars):
        """Also the hessian"""
        if (self.likelihood_vars is None or 
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        p, cov, chol, logLike = self.likelihood_vars
        
        # gradient #
        yyTmC  = self.yyT - cov
        dcov   = self.covariance_derivative(pars)
        icov   = alg.cho_solve(chol, np.identity(self.n))
        cidc   = [alg.blas.dsymm(1.0, icov, dc) for dc in dcov]
        cidcci = [alg.blas.dsymm(1.0, icov, ci, side=1) for ci in cidc]
        grad   = np.array([0.5 * np.trace(np.dot(yyTmC, ci)) for ci in cidcci])

        # hessian #
        npar   = len(grad)
        dcov2  = self.covariance_derivative2(pars)
        hess   = np.zeros((npar, npar))
        for i in range(npar):
            for j in range(i+1):
                cid2c      = alg.blas.dsymm(0.5, icov, dcov2[i,j])
                cid2cci    = alg.blas.dsymm(1.0, icov, cid2c, side=1)
                hess[i,j]  = 0.5 * np.trace(alg.blas.dsymm(1.0, cidc[i], cidc[j]))
                hess[i,j] += np.trace(np.dot(yyTmC, np.dot(cidc[i], cidcci[j]) - cid2cci))
                hess[j,i]  = hess[i,j]
        
        return logLike, grad, hess

    
    def sample(self, pars, size=1):
        cov  = self.covariance(pars)
        y = np.random.multivariate_normal(np.zeros(self.n), cov, size=size)
        return y
        

class FqLagBin(FqLagBase):
    """Base for all classes that fit bins in the frequency domain"""
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None):
        self.fql  = np.array(fql)
        self.fq   = (self.fql[1:] + self.fql[:-1])/2
        self.nfq  = len(self.fq)
        self.dt   = dt
        super(FqLagBin, self).__init__(tarr, yarr, yerr)
        if dt is None:
            self.calculate_integrals()
        else:
            self.calculate_integrals_aliasCorr()
        
    def calculate_integrals(self):
        
        tt  = np.array(self.tmat)
        fql = self.fql
        
        i0 = tt == 0
        tt[i0] = 1
        
        ang = 2 * np.pi * fql * tt[...,None]
        sin = np.sin(ang)
        I_s = (sin[...,1:] - sin[...,:-1]) / (2*np.pi * tt[...,None])
        I_s[i0] = (fql[1:] - fql[:-1])
        
        cos = np.cos(ang)
        I_c = (cos[...,1:] - cos[...,:-1]) / (2*np.pi * tt[...,None])
        I_c[i0] = 0

        # shape: n, n, nfq
        self.I_s = I_s
        self.I_c = I_c


    def calculate_integrals_aliasCorr(self):
        
        tt  = np.array(self.tmat)
        fql = self.fql
        dt  = self.dt
        
        
        # ang(3, n, n, nfq+1)
        ang   = 2 * np.pi * fql * (tt + np.array([-dt, 0, dt])[:,None,None])[...,None]
        si,ci = sp.sici(ang)
        # remove inf from ci; they are going to be multiplied by 0 anyway
        i0    = (tt == 0) | (np.abs(tt) == dt)
        #for i in range(3): ci[i][i0] = -1
        ci[0][(tt - dt) == 0] = -1
        ci[1][(tt     ) == 0] = -1
        ci[2][(tt + dt) == 0] = -1
        sin = np.sin(ang)
        cos = np.cos(ang)
        
        norm = 1./(4 * np.pi**2 * dt**2 * fql)
        I0_s = np.sum((ang * si + cos) * np.array([1, -2, 1])[:,None,None,None], 0) * norm
        I_s  = I0_s[...,1:] - I0_s[...,:-1]
        
        I0_c = np.sum((ang * ci - sin) * np.array([1, -2, 1])[:,None,None,None], 0) * norm
        I_c  = I0_c[...,1:] - I0_c[...,:-1]
        
        
        # shape: n, n, nfq
        self.I_s = I_s
        self.I_c = I_c


    