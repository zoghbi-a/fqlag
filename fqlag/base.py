
import numpy as np
import scipy.linalg as alg
import scipy.special as sp


class FqLagBase:
    """A base class for all frequency-based likelihood calculations

    This is not meant to be called directly.
    Calculations of covariances and likelihood functions is done here
    so we don't have to repeat it for psd and lag subcalsses.

    Args:
        tarr: a numpy array giving the time axis of the light curve.
        yarr: a numpy array giving the count rate or flux
        yerr: a numpy array giving the 1-sigma measurement uncertainity
            in the count rate or flux.

    Returns:
        Nothing

    Raises:
        ValueError if:
            - array shapes do not match

    """
    
    def __init__(self, tarr, yarr, yerr):
        """Initialize the FqLagBase class

        This is not meant to be called directly, but from subclasses 
        
        defines self.*:
            n:      light curve length
            tarr:   time array
            yarr:   mean-subtracted rate/flux array
            sig2:   the variance array corresponding to rate/flux
            mu:     light curve mean.
            tamt:   matrix of time differences (often refered to as time lags)
                that defines the auto and cross covariance.
            yyT:    yarr * yarr^T matrix used in the likelihood calculation
            likelihood_vars: a list of variables used for simple caching.

        """

        # A simple check for the input #
        tarr, yarr, yerr = np.array(tarr), np.array(yarr), np.array(yerr)
        if not(len(tarr) == len(yarr) == len(yerr)):
            raise ValueError('Input arrays must be equal in size')

        # initialize simple parameters #
        self.n    = tarr.shape[0]
        self.tarr = tarr
        self.mu   = yarr.mean()
        self.yarr = yarr - self.mu
        self.sig2 = yerr**2
        self.tmat = tarr - np.expand_dims(tarr, 1)
        self.yyT  = self.yarr * np.expand_dims(self.yarr, 1)
        self.likelihood_vars = None
        
    
    def covariance(self, pars):
        """Covariance kernel function. 
        
        This is meant to be overloaded by child classes
        
        Args:
            pars: parameters that defined the covariance kernel.

        Returns:
            A covariance matrix of shape: (self.n, self.n)

        """

        # dummy function, child classes should overload this
        # as an example, we use a linear functions:
        # cov (tau) = pars[0] * tau + pars[1]
        return np.abs(self.tmat) * pars[0] + pars[1]
    

    def covariance_derivative(self, pars):
        """First Derivative of the covariance kernel function
        
        This is also meant to be overloaded by child classes.

        Args:
            pars: parameters of the covariance kernel.

        Returns:
            a matrix of first derivatives with shape (npar, self.n, self.n)

        """
        # for the dummpy example of a linear functions, we have:
        # d_cov/d_par[0] = tau
        # d_cov/d_par[1] = 1
        return np.array([
            np.abs(self.tmat),
            np.abs(self.tmat)*0 + 1
        ])


    def covariance_derivative2(self, pars):
        """Second Derivative of the covariance kernel function

        This is also meant to be overloaded by child classes.

        Args:
            pars: parameters of the covariance kernel.

        Returns:
            a matrix of second derivatives with shape (npar, npar, self.n, self.n)
        """
        # for the dummpy example of a linear functions, we have:
        # all second derivatives are zero
        return np.array([[self.tmat*0, self.tmat*0],
                         [self.tmat*0, self.tmat*0]])

    
    def loglikelihood(self, pars):
        """Calculate the log of the likelihood function.

        This function calls self.covariance to construct a covariance
        matrix. After adding a diagonal measurement noise, the matrix
        is factored and inverted (implicitely) and likelihood is calculated.


        Args:
            pars: parameters of the model as a numpy array

        Returns:
            a single number giving the log-likelihood

        Raises:
            LinAlgError: if the covariance is singular and cannot be factored

        """
        # calculate covariance matrix #
        cov = self.covariance(pars)

        # add measurement noise; considering independent gaussian noise only #
        cov += np.diag(self.sig2)

        # factor the covariance and calculate the loglikelihood #
        # this is the heavy part of the code 
        chol = alg.cho_factor(cov, lower=False)
        logDet = 2 * np.sum(np.log(np.diag(chol[0])))
        chi2 = np.dot(self.yarr, alg.cho_solve(chol, self.yarr))
        logLike = -0.5 * ( chi2 + logDet + self.n*np.log(2*np.pi) )

        # simple caching, in case we call the derivative with the same pars #
        self.likelihood_vars = [np.array(pars), cov, chol, logLike]

        return logLike
    

    def loglikelihood_derivative(self, pars, calc_fisher=True):
        """Calculate the first derivative of the log of the likelihood function.

        The calculation follows from Bond et al. 1998 
        (https://ui.adsabs.harvard.edu/abs/1998PhRvD..57.2117B/abstract)

        
        Args:
            pars: parameters of the model as a numpy array
            calc_fisher: Calculate the fisher matrix to approximate the
                the Hessian. This may or not be needed depending on the
                optimizition algorithm used. Default is True.

        Returns:
            if calc_fisher is True:
                return logLikelihood, gradient_array
            else:
                return: logLikelihood, gradient_array, fisher_matrix

        Raises:
            LinAlgError: if the covariance is singular and cannot be factored

        """

        # use the cached variables if loglikelihood has already been called #
        if (self.likelihood_vars is None or 
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        p, cov, chol, logLike = self.likelihood_vars
        
        # calculate the gradient #
        yyTmC  = self.yyT - cov
        dcov   = self.covariance_derivative(pars)
        icov   = alg.cho_solve(chol, np.identity(self.n))
        cidc   = [alg.blas.dsymm(1.0, icov, dc) for dc in dcov]
        cidcci = [alg.blas.dsymm(1.0, icov, ci, side=1) for ci in cidc]
        grad   = np.array([0.5 * np.trace(np.dot(yyTmC, ci)) for ci in cidcci])
        
        # if Fisher matrix is not needed, return here #
        if not calc_fisher:
            return logLike, grad


        # Fisher matrix, approximate Hessian #
        npar   = len(grad)
        fisher = np.zeros((npar, npar))
        for i in range(npar):
            for j in range(i+1):
                fisher[i,j] = 0.5 * np.trace(alg.blas.dsymm(1.0, cidc[i], cidc[j]))
                fisher[j,i] = fisher[i,j]
        return logLike, grad, fisher


    def loglikelihood_derivative2(self, pars):
        """Calculate the second derivative of the log of the likelihood function.

        The calculation follows from Bond et al. 1998 
        (https://ui.adsabs.harvard.edu/abs/1998PhRvD..57.2117B/abstract)

        
        Args:
            pars: parameters of the model as a numpy array

        Returns:
            logLikelihood, gradient_array, Hessian_matrix

        Raises:
            LinAlgError: if the covariance is singular and cannot be factored

        """
        # use the cached variables if loglikelihood has already been called #
        if (self.likelihood_vars is None or 
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        p, cov, chol, logLike = self.likelihood_vars
        
        # calculate the gradient #
        yyTmC  = self.yyT - cov
        dcov   = self.covariance_derivative(pars)
        icov   = alg.cho_solve(chol, np.identity(self.n))
        cidc   = [alg.blas.dsymm(1.0, icov, dc) for dc in dcov]
        cidcci = [alg.blas.dsymm(1.0, icov, ci, side=1) for ci in cidc]
        grad   = np.array([0.5 * np.trace(np.dot(yyTmC, ci)) for ci in cidcci])

        
        # calculate the Hessian #
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
        """Generate random light curves given covariance parameters pars

        Args:
            pars: parameters of the model as a numpy array
            size: how many samples to generate. Default: 1
        
        Returns:
            an array of size (size, n)

        """

        # calculate the covariance matrix #
        cov  = self.covariance(pars)

        # generate the random variates #
        y = np.random.multivariate_normal(np.zeros(self.n), cov, size=size)

        return y
        

class FqLagBin(FqLagBase):
    """A base class for all models that fit bins in the frequency-domain.

    This is not meant to be called directly.
    This classes primarily handles the Fourier integrals that are common
    to models that divides the frequency axis into bins.

    Args:
        tarr: a numpy array giving the time axis of the light curve.
        yarr: a numpy array giving the count rate or flux
        yerr: a numpy array giving the 1-sigma measurement uncertainity
            in the count rate or flux.
        fql: a numpy array of frequency bin boundaries
        dt: sampling time of the light curves. If given, corrections to sampling
        bias is applied, otherwise, we don't apply it.

    Returns:
        Nothing

    """
    
    def __init__(self, tarr, yarr, yerr, fql, dt=None):
        """Initialize the FqLagBin class

        This is not meant to be called directly, but from subclasses 
        
        defines self.* (in addition to those defined in FqLagBase):
            fql:    array of frequency boundaries
            fq:     the geometric center of the frequency bins
            nfq:    Number of frequency bins
            sig2:   the variance array corresponding to rate/flux
            dt:     sampling time (if bias correction is needed, else None)

        """
        self.fql  = np.array(fql)
        self.fq   = np.exp((np.log(self.fql[1:])+np.log(self.fql[:-1]))/2.0)
        self.nfq  = len(self.fq)
        self.dt   = dt
        super(FqLagBin, self).__init__(tarr, yarr, yerr)
        if dt is None:
            self.calculate_integrals()
        else:
            self.calculate_integrals_aliasCorr()
        

    def calculate_integrals(self):
        """Calculate the Fourier integrals binned frequency axis

        These integrals do not depend on the model parameters, so they
        can be pre-calculated at the model initialization.
        The integrals are stored in self.I_s and self.I_c

        with:
        I_s = int_{f1}^{f2}{Cos(2 pi f tau) df}
        I_c = int_{f1}^{f2}{Sin(2 pi f tau) df}
        with f1,f2 being the bin boundaries.

        Args:
            None

        Returns:
            None

        """
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

        # shape: (n, n, nfq)
        self.I_s = I_s
        self.I_c = I_c


    def calculate_integrals_aliasCorr(self):
        """Calculate the Fourier integrals binned frequency axis 
        with aliasing correction

        These integrals do not depend on the model parameters, so they
        can be pre-calculated at the model initialization.
        The integrals are stored in self.I_s and self.I_c

        with:
        I_s = int_{f1}^{f2}{Cos(2 pi f tau) Sinc(pi f dt)^2 df}
        I_c = int_{f1}^{f2}{Sin(2 pi f tau) Sinc(pi f dt)^2 df}
        with f1,f2 being the bin boundaries.

        Args:
            None

        Returns:
            None

        """
        tt  = np.array(self.tmat)
        fql = self.fql
        dt  = self.dt
        
        
        # ang.shape = (3, n, n, nfq+1)
        ang   = 2 * np.pi * fql * (tt + np.array([-dt, 0, dt])[:,None,None])[...,None]
        si,ci = sp.sici(ang)
        # remove inf from ci; they are going to be multiplied by 0 anyway
        i0    = (tt == 0) | (np.abs(tt) == dt)
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
        
        
        # shape: (n, n, nfq)
        self.I_s = I_s
        self.I_c = I_c

    
