"""Base class for all frequency-based likelihood calculations"""

import numpy as np
import scipy.linalg as alg
import scipy.special as sp
import scipy.stats as st


class FqLagBase:
    """A base class for all frequency-based likelihood calculations

    This is not meant to be called directly.
    Calculations of covariances and likelihood functions is done here
    so we don't have to repeat it for psd and lag subcalsses.

    """

    def __init__(self,
                 tarr: np.ndarray,
                 yarr: np.ndarray,
                 yerr: np.ndarray):
        """Initialize the FqLagBase class

        This is not meant to be called directly, but from subclasses 

        Parameters
        ----------
        tarr: np.ndarray
            a numpy array giving the time axis of the light curve.
        yarr: np.ndarray
            a numpy array giving the count rate or flux
        yerr: np.ndarray
            a numpy array giving the 1-sigma measurement uncertainity
            in the count rate or flux.

        defines self.*:
            npoints: light curve length
            tarr:    time array
            yarr:    mean-subtracted rate/flux array
            sig2:    the variance array corresponding to rate/flux
            mean:    light curve mean.
            tamt:    matrix of time differences (often refered to as time lags)
                that defines the auto and cross covariance.
            yy_t:    yarr * yarr^T matrix used in the likelihood calculation
            likelihood_vars: a list of variables used for simple caching.

        """

        # A simple check for the input #
        tarr, yarr, yerr = np.array(tarr), np.array(yarr), np.array(yerr)
        if not len(tarr) == len(yarr) == len(yerr):
            raise ValueError('Input arrays must be equal in size')

        # initialize simple parameters #
        self.npoints = tarr.shape[0]
        self.tarr = tarr
        self.mean   = yarr.mean()
        self.yarr = yarr - self.mean
        self.sig2 = yerr**2
        self.tmat = tarr - np.expand_dims(tarr, 1)
        self.yy_t  = self.yarr * np.expand_dims(self.yarr, 1)
        self.likelihood_vars = None
        self.params = {}
        self.priors = None


    def covariance(self, pars: np.ndarray):
        """Covariance kernel function. 
        
        This is meant to be overloaded by child classes
        
        Parameters
        ----------
        pars: np.ndarray
            parameters that define the covariance kernel.

        Returns
        -------
        A covariance matrix of shape: (self.npoints, self.npoints)

        """

        # dummy function, child classes should overload this
        # as an example, we use a linear functions:
        # cov (tau) = pars[0] * tau + pars[1]
        return np.abs(self.tmat) * pars[0] + pars[1]


    def covariance_derivative(self, pars: np.ndarray):
        """First Derivative of the covariance kernel function
        
        This is also meant to be overloaded by child classes.

        Parameters
        ----------
        pars: np.ndarray
            parameters that define the covariance kernel.

        Returns
        -------
        a matrix of first derivatives with shape (npar, self.npoints, self.npoints)

        """
        # for the dummpy example of a linear functions, we have:
        # d_cov/d_par[0] = tau
        # d_cov/d_par[1] = 1
        _ = pars
        return np.array([
            np.abs(self.tmat),
            np.abs(self.tmat)*0 + 1
        ])


    def covariance_derivative2(self, pars: np.ndarray):
        """Second Derivative of the covariance kernel function

        This is also meant to be overloaded by child classes.

        Parameters
        ----------
        pars: np.ndarray
            parameters that define the covariance kernel.

        Returns
        -------
        a matrix of second derivatives with shape (npar, npar, self.npoints, self.npoints)

        """
        _ = pars
        # for the dummpy example of a linear functions, we have:
        # all second derivatives are zero
        return np.array([[self.tmat*0, self.tmat*0],
                         [self.tmat*0, self.tmat*0]])


    def loglikelihood(self, pars: np.ndarray):
        """Calculate the log of the likelihood function.

        This function calls self.covariance to construct a covariance
        matrix. After adding a diagonal measurement noise, the matrix
        is factored and inverted (implicitely) and likelihood is calculated.


        Parameters
        ----------
        pars: np.ndarray
            parameters of the model as an array

        Returns
        -------
        a float giving the value of the log-likelihood.

        Raises
        ------
        LinAlgError: if the covariance is singular and cannot be factored

        """
        # calculate covariance matrix #
        cov = self.covariance(pars)

        # add measurement noise; considering independent gaussian noise only #
        cov += np.diag(self.sig2)

        # factor the covariance and calculate the log_likelihood #
        # this is the heavy part of the code
        chol = alg.cho_factor(cov, lower=False)
        log_det = 2 * np.sum(np.log(np.diag(chol[0])))
        chi2 = np.dot(self.yarr, alg.cho_solve(chol, self.yarr))
        log_like = -0.5 * ( chi2 + log_det + self.npoints*np.log(2*np.pi) )

        # add priors:
        if self.priors is not None:
            for ipr,prr in self.priors.items():
                log_like += prr.logpdf(pars[ipr])

        # simple caching, in case we call the derivative with the same pars #
        self.likelihood_vars = [np.array(pars), cov, chol, log_like]

        return log_like


    def loglikelihood_derivative(self,
                                 pars: np.ndarray,
                                 calc_fisher: bool = True):
        """Calculate the first derivative of the log of the likelihood function.

        The calculation follows from Bond et al. 1998 
        (https://ui.adsabs.harvard.edu/abs/1998PhRvD..57.2117B/abstract)


        Parameters
        ----------
        pars: np.ndarray
            parameters of the model as an array
        calc_fisher: bool
            Calculate the fisher matrix to approximate the
            the Hessian. This may or not be needed depending on the
            optimizition algorithm used. Default is True.


        Returns
        -------
        if calc_fisher is True:
            return loglikelihood, gradient_array
        else:
            return: log_likelihood, gradient_array, fisher_matrix

        Raises
        ------
        LinAlgError: if the covariance is singular and cannot be factored

        """

        # use the cached variables if loglikelihood has already been called #
        if (self.likelihood_vars is None or
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        _, cov, chol, log_like = self.likelihood_vars

        # calculate the gradient #
        yy_tmc = self.yy_t - cov
        dcov   = self.covariance_derivative(pars)
        icov   = alg.cho_solve(chol, np.identity(self.npoints))
        cidc   = [alg.blas.dsymm(1.0, icov, dc) for dc in dcov] # pylint: disable=no-member
        cidcci = [alg.blas.dsymm(1.0, icov, ci, side=1) for ci in cidc] # pylint: disable=no-member
        grad   = np.array([0.5 * np.trace(np.dot(yy_tmc, ci)) for ci in cidcci])

        # if Fisher matrix is not needed, return here #
        if not calc_fisher:
            return log_like, grad


        # Fisher matrix, approximate Hessian #
        npar   = len(grad)
        fisher = np.zeros((npar, npar))
        for i in range(npar):
            for j in range(i+1):
                fisher[i,j] = 0.5 * np.trace(alg.blas.dsymm(1.0, cidc[i], cidc[j]))
                fisher[j,i] = fisher[i,j]
        return log_like, grad, fisher


    def loglikelihood_derivative2(self, pars: np.ndarray):
        """Calculate the second derivative of the log of the likelihood function.

        The calculation follows from Bond et al. 1998 
        (https://ui.adsabs.harvard.edu/abs/1998PhRvD..57.2117B/abstract)

        
        Parameters
        ----------
        pars: np.ndarray
            parameters of the model as an array

        Returns
        -------
        log_likelihood, gradient_array, Hessian_matrix

        Raises
        ------
        LinAlgError: if the covariance is singular and cannot be factored

        """
        # use the cached variables if loglikelihood has already been called #
        if (self.likelihood_vars is None or
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        _, cov, chol, log_like = self.likelihood_vars

        # calculate the gradient #
        yy_tmc  = self.yy_t - cov
        dcov   = self.covariance_derivative(pars)
        icov   = alg.cho_solve(chol, np.identity(self.npoints))
        cidc   = [alg.blas.dsymm(1.0, icov, dc) for dc in dcov]
        cidcci = [alg.blas.dsymm(1.0, icov, ci, side=1) for ci in cidc]
        grad   = np.array([0.5 * np.trace(np.dot(yy_tmc, ci)) for ci in cidcci])


        # calculate the Hessian #
        npar   = len(grad)
        dcov2  = self.covariance_derivative2(pars)
        hess   = np.zeros((npar, npar))
        for i in range(npar):
            for j in range(i+1):
                cid2c      = alg.blas.dsymm(0.5, icov, dcov2[i,j])
                cid2cci    = alg.blas.dsymm(1.0, icov, cid2c, side=1)
                hess[i,j]  = 0.5 * np.trace(alg.blas.dsymm(1.0, cidc[i], cidc[j]))
                hess[i,j] += np.trace(np.dot(yy_tmc, np.dot(cidc[i], cidcci[j]) - cid2cci))
                hess[j,i]  = hess[i,j]

        return log_like, grad, hess


    def add_prior(self,
                  ipar: int,
                  prior: st._distn_infrastructure.rv_continuous_frozen):
        """Add a prior on a parameter

        Parameters
        ----------
        ipar: int
            parameter number for which to apply the prior (0 based)
        prior: scipy.stats._distn_infrastructure.rv_continuous_frozen 
            Defines the probability distribution for the prior.
            for example: scipy.stats.norm(loc, scale) etc.

        """

        if not isinstance(prior, st._distn_infrastructure.rv_continuous_frozen):
            raise ValueError(('prior has to be instance of '
                              'scipt.stats._distn_infrastructure.rv_continuous_frozen'))
        self.priors = {ipar:prior}


    def sample(self,
               pars: np.ndarray,
               size: int = 1):
        """Generate random light curves given covariance parameters pars

        Parameters
        ----------
        pars: np.ndarray
            parameters of the model as a numpy array
        size: int
            how many samples to generate. Default: 1
        
        Returns
        -------
        An array of size (size, n)

        """

        # use the cached variables if loglikelihood has already been called #
        if (self.likelihood_vars is None or
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        cov = self.likelihood_vars[1]

        # generate the random variates #
        return  np.random.multivariate_normal(np.zeros(self.npoints), cov, size=size)


    def conditional_predict(self,
                            pars: np.ndarray,
                            tnew: np.ndarray,
                            sample: int = None,
                            **kwargs):
        """Predict the values at times tnew give the parameter pars

        From page 16 in Rasmussen & C. K. I. Williams:
            Gaussian Processes for Machine Learning. See
            also Zu, Kochanek, Peterson 2011

        Parameters
        ----------
        pars: np.ndarray
            parameters of the model as a numpy array
        tnew: np.ndarray
            array of times where the predictions are to be made.
        sample: int
            if not None, gives the number of random samples to be generated
            that correspond to new times tnew

        Keywords
        --------
        seed: int
            random seed in case sample is not None
        
        Returns
        -------
        if sample is None:
            (rarrNew, rerrNew): Light curve estimates rarrNew and their uncertainties 
            rerrNew at the times tnew
        else:
            (rarrNew, rerrNew, samples), giving additionally randomly generated samples

        """
        seed = kwargs.get('seed', None)
        np.random.seed(seed)

        n_d  = self.npoints
        mean = self.mean

        # augmented arrays
        t_aug = np.concatenate((self.tarr, tnew))
        r_aug = np.zeros_like(t_aug) + mean
        # the error is not needed here.
        newmod = type(self)(t_aug, r_aug, r_aug, **self.params)



        # use the cached variables if loglikelihood has already been called #
        if (self.likelihood_vars is None or
                not np.all(np.isclose(pars, self.likelihood_vars[0]))):
            self.loglikelihood(pars)
        _, cov, chol, log_like = self.likelihood_vars
        Ci   = alg.cho_solve(chol, np.identity(n_d))
        Ciy  = np.dot(Ci, self.yarr)

        # covariance of the new model
        S     = newmod.covariance(pars)
        S_sd  = S[:n_d, n_d:].T
        S_ss  = S[n_d:, n_d:]

        # new values and their uncertainties
        rarrNew = np.dot(S_sd, Ciy) + mean
        ycov = S_ss - np.dot(np.dot(S_sd, Ci), S_sd.T)
        rerrNew = np.sqrt(np.diag(ycov))


        # generate the random variates if requested #
        if not sample is None:
            sample  = np.int(sample)
            samples = np.random.multivariate_normal(rarrNew, ycov, size=sample)
            return rarrNew, rerrNew, samples

        return rarrNew, rerrNew


class FqLagBin(FqLagBase):
    """A base class for all models that fit bins in the frequency-domain.

    This is not meant to be called directly.
    This classes primarily handles the Fourier integrals that are common
    to models that divides the frequency axis into bins.

    """

    def __init__(self,
                 tarr: np.ndarray,
                 yarr: np.ndarray,
                 yerr: np.ndarray,
                 fql: np.ndarray,
                 dt: float = None):
        """Initialize the FqLagBin class

        This is not meant to be called directly, but from subclasses 
        
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
        super().__init__(tarr, yarr, yerr)
        if dt is None:
            self.calculate_integrals()
        else:
            self.calculate_integrals_aliasCorr()
        self.params = {'fql':fql, 'dt':dt}


    def calculate_integrals(self):
        """Calculate the Fourier integrals binned frequency axis

        These integrals do not depend on the model parameters, so they
        can be pre-calculated at the model initialization.
        The integrals are stored in self.I_s and self.I_c

        with:
        I_s = int_{f1}^{f2}{Cos(2 pi f tau) df}
        I_c = int_{f1}^{f2}{Sin(2 pi f tau) df}
        with f1,f2 being the bin boundaries.

        """
        tt  = np.array(self.tmat)
        fql = self.fql

        i0 = tt == 0
        tt[i0] = 1

        ang = 2 * np.pi * fql * tt[...,None]
        sin = np.sin(ang)
        I_s = (sin[...,1:] - sin[...,:-1]) / (2*np.pi * tt[...,None])
        I_s[i0] = fql[1:] - fql[:-1]

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

        """
        tt  = np.array(self.tmat)
        fql = self.fql
        dt  = self.dt


        # ang.shape = (3, n, n, nfq+1)
        dt = np.double(dt)
        ang   = 2. * np.pi * fql * (tt + np.array([-dt, 0.0, dt])[:,None,None])[...,None]
        si,ci = sp.sici(ang)
        # remove inf from ci; they are going to be multiplied by 0 anyway
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
