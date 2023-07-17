"""Misc utilty methods"""
# pylint: skip-file
import emcee
import numpy as np
import scipy.optimize as opt
import scipy.stats as st
from scipy.misc import derivative


def check_grad(mod, p0, dx=1e-3):
    """Compare the gradient from mod.loglikelihood_derivative
    against numerical derivative.

    Tests that the derivatie codes are correct

    Parameters
    ----------
    mod: FqLagBase
        the model we are testing
    p0: np.ndarray
        parameters of the model
    dx: float
        used for the numerical derivative

    Returns
    -------
    logLikelihood, grad_array_analytic, grad_array_numerical
    """

    p0 = np.array(p0)
    l,g,_ = mod.loglikelihood_derivative(p0)

    g0 = []
    for i,_ in enumerate(p0):
        def f(x, p0, ii):
            p0[ii] = x
            return mod.loglikelihood(p0)
        g0.append(derivative(f, p0[i], dx, args=(np.array(p0), i)))
    g0 = np.array(g0)

    print('analytic:  ', ' '.join([f'{x:10.4}' for x in g]))
    print('numerical: ', ' '.join([f'{x:10.4}' for x in g0]))
    return l, g, g0


def run_mcmc(mod, p0, perr=None, nwalkers=-10, nrun=100, **kwargs):
    """Run MCMC to estimate the Posterior distributions of the parameters

    This uses the emcee package.

    Parameters
    ----------
    mod: FqLagBase
        the model object for calulating the likelihood function
    p0: np.ndarray
        starting paramters for chain
    perr: np.ndarray
        estimated uncertainties on the parameters. They are used to 
        initialize the chains. If not given, the chain is started with 
        values within 10% of p0
    nwalkers: int
        number of walkers in the chains (see emcee for details)
    nrun: int
        number of chain runs

    Keywords
    --------
    sigma_f: float
        the factor that multiples perr used to initialize the 
        walkers. Default: 0.5
    limits: list
        a list of [pmin, pmax] values for the limits on the parameters.
        These are effectively used as uniform priors on the parameters
    iphi: list of int
        The indicies of the phase parameters within p0. Used to ensure
        that those parameters are cyclic and remain -pi < phi < pi

    Returns
    -------
    the chain array where the walker axis is flattened.

    """

    sigma_f = kwargs.get('sigma_f', 0.5)
    limits  = kwargs.get('limits', None)
    iphi    = kwargs.get('iphi', None)
    if limits is None:
        limits = [[-30,30] for x in p0]
    if iphi is None:
        iphi = []

    def log_prob(x, mod):
        for ix,_ in enumerate(x):
            if ix in iphi:
                x[ix] = (x[ix]+np.pi) % (2*np.pi) - np.pi
            if x[ix]<limits[ix][0] or x[ix]>limits[ix][1]:
                return -np.inf
        #if np.any(np.logical_or(x < -30, x > 30)):
        #    return -np.inf
        try:
            l = mod.loglikelihood(x)
        except np.linalg.LinAlgError:
            l = -np.inf
        return l

    ndim = len(p0)
    if nwalkers < 0:
        nwalkers = -ndim * nwalkers
    pe = p0 * 0.1 if perr is None else perr

    p0      = np.random.randn(nwalkers, ndim)*pe*sigma_f + p0
    p0 = np.array([[np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)] for x in p0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[mod,])
    _ = sampler.run_mcmc(p0, nrun)
    pchain  = sampler.flatchain
    lchain  = sampler.flatlnprobability
    print('acceptance fraction: ', np.mean(sampler.acceptance_fraction))
    chain   = np.hstack([pchain, np.expand_dims(lchain, -1)])
    return chain


def maximize(mod, p0, limits=None, ipfix=None, useGrad=False, **kwargs):
    """Maixmize the likelihood of model mod

    Use numerical optimization from scipy.optimize.minimize to estimate
    the parameters of the model at the likelihood maximum
    We the BFGS algorithm.

    Parameters
    ----------
    mod: FqLagBase
        model whose likelihood is to be optimized
    p0: np.ndarray
        starting model parameters
    limits: list
        a list of [pmin, pmax] values for the limits on the parameters.
        These are effectively used as uniform priors on the parameters.
        None means all parameters are assumed to be between [-30, 30]
    ipfix: list of in
        parameter indices of p0 to keep fixed during the maximization.
        Useful when calculating uncertainties by stepping through them.
    useGrad: bool
        use analytical gradient. This may give ~10% speedup, but it can be
        unstable for complex problems.

    Keywords
    --------
    verbose: bool
        if True, print progress

    Returns
    -------
    return (pars_best, pars_best_error, fit_result) 
        the latter is from scipy.optimize.minimize

    """
    verbose = kwargs.get('verbose', True)

    if limits is None:
        limits = [[-30,30] for x in p0]

    if ipfix is None:
        ipfix = []
    npar = len(p0)
    pfix = np.array([p0[i] for i in ipfix])
    ivar = [i for i in range(npar) if not i in ipfix]
    info = [npar, pfix, ipfix, ivar]

    # main negative log-likelihood function #
    def f(x, mod, info):
        npar, pfix, ipfix, ivar = info
        x = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)])
        y = np.zeros(npar, np.double)
        y[ipfix] = pfix
        y[ivar ] = x
        #y = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(y,limits)])

        try:
            l = mod.loglikelihood(y)
        except np.linalg.LinAlgError:
            l = -1e6

        if verbose and not useGrad:
            print(f'{l:10.6} | ', ' '.join([f'{xx:10.3}' for xx in x]), '\r', end='')

        return -l

    # first derivative of the negative log-likelihood
    def fprime(x, mod, info):
        npar, pfix, ipfix, ivar = info
        x = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)])
        y = np.zeros(npar, np.double)
        y[ipfix] = pfix
        y[ivar ] = x

        try:
            l, g = mod.loglikelihood_derivative(y, calc_fisher=False)
            g = g[ivar]
        except np.linalg.LinAlgError:
            l = -1e6
            g = x*0 - 1e6
        if verbose:
            print(f'{l:10.6} | ', ' '.join([f'{xx:10.3}' for xx in x]),
                  f'{np.max(np.abs(g))}\r', end='')

        return -g

    if not useGrad:
        fprime = None
    res = opt.minimize(f, p0[ivar], args=(mod, info), method='BFGS', tol=1e-4, jac=fprime,
                options={'gtol':1e-4})

    # last print
    if verbose:
        print(f'{-res.fun:10.6} | ', ' '.join([f'{xx:10.3}' for xx in res.x]),
              f' | {np.max(np.abs(res.jac)):10.3}\r', end='')
        print('\n** done **\n')

    p, pe = res.x, np.diag(res.hess_inv)**0.5
    y, ye = np.zeros(npar, np.double), np.zeros(npar, np.double)
    y[ipfix] = pfix
    y[ivar ] = p
    ye[ivar] = pe
    y = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(y,limits)])

    return y, ye, res


def step_par(mod, p0, par1, par2=None, **kwargs):
    """Step a parameter thorugh an array and record the change in the
    likelihood function, fitting other parameters each time.

    It can be used to calculate the uncertainties of some parameters

    Parameters
    ----------
    mod: FqLagBase
        model whose likelihood is to be optimized
    p0: np.ndarray
        starting model parameters
    par1: list
        [ipar, p_array], where ipar is the index of the parameter
        in p0 to step through, and p_array is the array of parameters
        to use
    par1: list
        similar to par1 to do two parameters. Default is None, so we 
        only do one parameter

    Keywords
    --------
    verbose: bool
        if True, print progress
    limits: list
        a list of [pmin, pmax] values for the limits on the parameters.
        to be passed to @maximize

    Returns
    -------
    step, [pbest, pbest_e, lbest] where:
    step: (n, 2) array with parameter value and loglikelihood
    pbest, pbest_e, lbest: parameter list, errors and the best loglikelihood
        values.

    """

    verbose = kwargs.get('verbose', True)
    limits  = kwargs.get('limits', None) # used for maximize

    # find best fit first
    pbest, pbest_e, res = maximize(mod, p0, limits, verbose=False)
    lbest = -res.fun
    if verbose:
        print(f'best loglikelihood: {lbest:10.6}')

    ip1 = par1[0]
    step = []
    for _,p1 in enumerate(par1[1]):
        p = np.array(pbest)
        p[ip1] = p1
        if not par2 is None:
            ip2 = par2[0]
            step2 = []
            for _,p2 in enumerate(par2[1]):
                pp = np.array(p)
                pp[ip2] = p2
                res = maximize(mod, pp, limits, ipfix=[ip1, ip2], verbose=False)
                step2.append([p1, p2, -res[2].fun])
                if verbose:
                    print(f'{step2[-1][0]:10.3} {step2[-1][1]:10.3} {step2[-1][2]:10.6} ',
                          f'{np.round(lbest - step2[-1][-1], 2):10.3}\r', end='')
            step.append(step2)
        else:
            res = maximize(mod, p, limits, ipfix=[ip1], verbose=False)
            step.append([p1, -res[2].fun])
            if verbose:
                print(f'{step[-1][0]:10.3} {step[-1][1]:10.6} '
                      f'{np.round(lbest - step[-1][-1], 2):10.3}\r', end='')
    step = np.array(step)
    return step, [pbest, pbest_e, lbest]


def errors(mod, p0, ipars=None, **kwargs):
    """Calculate the uncertainties in the parameters p0 that maximize
    the likelihood function of a model mod.

    For each parameter, the value is changed in small steps until the log-likelihood
    changes by DCHI2 (default is 1, to calculated the 1-sigma uncertainties)

    Parameters
    ----------
    mod: FqLagBase
        model whose log-likelihood can called as mod.loglikelihood
    p0: np.ndarray
        the model parameters that maximize the likelihood, obtained 
        for example by running @misc.maximize
    ipars: list of int
        a list of indices of p0 for which the errors are to be calculated
        Default: None, means calculate errors for all parameters


    Keywords
    --------
    limits: list
        a list of [pmin, pmax] values for the limits on the parameters.
        to be passed to @maximize
    tol: float
        tolerance in loglikelihood value. e.g. calculation stop when
        |Delta(loglikelihood) - DCHI2| < tol. Default: 1e-2
    DCHI2: float
        the change in loglikelihood value to probe. DCHI2=1 gives ~1-sigma
        uncertainties. For 90% confidence for instance, use DCHI2=2.71.
    skip_ipars: list
        Parameter indices to skip in calculating the errors. Usefull
        for example in combination with ipars=None above.
    sign: int
        the direction of the parameter uncertainty search. Default:1 means
        increase the parameter until Delta(loglikelihood)=DCHI2. -1 means seach
        in the other direction. Doing one direction assumes gaussian uncertainties.
        If the assumption breaks, both both +1 and -1 uncertainties should be reported.
    verbose: bool
        True to print progress


    """
    # limits on the parameters; act like uniform priors
    limits  = kwargs.get('limits', None)

    # tolerance #
    tol     = kwargs.get('tol', 1e-2)

    # a measure of the confidence level in the uncertainty.
    DCHI2   = kwargs.get('DCHI2', 1.0)

    # parameter indices to skip
    skip_ipars = kwargs.get('skip_ipars', [])

    # direction search for uncertainties.
    sign    = kwargs.get('sign', 1)

    # printing progress?
    verbose = kwargs.get('verbose', True)


    # do all parameters if ipars=None
    npar = len(p0)
    if ipars is None:
        ipars = list(range(npar))

    ipars = list(ipars)

    # make sure we start the search from parameters that maximize the likelihood
    pbest, pbest_e, res = maximize(mod, np.array(p0), limits)
    lbest = -res.fun

    # loop through the parameters. If a new maximum is found,
    # restart the whole search
    p, pe = np.array(pbest), np.array(pbest_e)
    for iipar, ipar in enumerate(ipars):

        if iipar in skip_ipars:
            continue
        if verbose:
            print(f'\t## errors for param {ipar} ##')

        # make sure DlogL is enclosed in the search region,
        # which is defined by integer multiples of pe
        isig, pExtrm, dchi2 = 0.5, p[ipar], 0
        not_bound = False
        while (dchi2 - DCHI2) < tol*2:
            pExtrm = p[ipar] + isig*sign*pe[ipar]
            tmpp   = np.array(p)
            tmpp[ipar] = pExtrm
            tmp_res = maximize(mod, tmpp, limits, ipfix=[ipar], verbose=False)
            dchi2 = 2*(lbest - (-tmp_res[2].fun))
            if dchi2 < (-2*tol):
                if verbose:
                    print('@@ a new best fit is found looping ... @@')
                    print(f'{-tmp_res[2].fun:10.6} | ',
                          ' '.join([f'{xx:10.3g}' for xx in tmp_res[0]]),'\r', end='')
                #ipars = ipars[iipar:] + ipars[:iipar]
                return errors(mod, tmp_res[0], ipars, **kwargs)
            isig += 0.5
            if isig >= 10:
                print((f'parameter {ipar} appears to be unbound using sign={sign}\n'
                         f'Try using sign={-sign}'))
                pbest_e[ipar] = np.abs(pExtrm - pbest[ipar])
                not_bound = True
                break
        if not_bound:
            continue
        # -------------------------------------------------- #


        ## change tmpp[ipar] until dchi2==DCHI2 with tolerence TOL ##
        pExtrm2 = p[ipar]
        icount = 0
        while np.abs(dchi2-DCHI2)>=tol:
            icount += 1
            pHalf = (pExtrm + pExtrm2)/2.
            tmpp[ipar] = pHalf
            tmp_res = maximize(mod, tmpp, limits, ipfix=[ipar], verbose=False)
            dchi2 = 2*(lbest - (-tmp_res[2].fun))
            if dchi2 < (-2*tol):
                if verbose:
                    print('@@ a new best fit is found looping ... @@')
                    print(f'{-tmp_res[2].fun:10.6} | ',
                          ' '.join([f'{xx:10.3g}' for xx in tmp_res[0]]),'\r', end='')
                #ipars = np.concatenate([ipars[iipar:], ipars[:iipar]])
                return errors(mod, tmp_res[0], ipars, **kwargs)
            if verbose:
                print(f' {lbest:10.6} {-tmp_res[2]:10.6} {p[ipar]:10.6}',
                      f' {pHalf:10.6} {dchi2:10.6}\r', end='')
            if dchi2 < DCHI2:
                pExtrm2 = pHalf
            else:
                pExtrm = pHalf
            if icount >= 50:
                break
        pbest_e[ipar] = np.abs(pHalf - pbest[ipar])
        print()

    # get fit result again, so the return is consistent with the return of maximize
    res = maximize(mod, pbest, limits, verbose=False)

    # finally, if DCHI2 != 1; scale the errors, so the return always corresponds to 1-sigma errors
    error_scale = (st.norm.ppf((1-st.chi2.cdf(1, 1))/2) / st.norm.ppf((1-st.chi2.cdf(DCHI2, 1))/2))
    pbest_e[ipars] = pbest_e[ipars] * error_scale

    print('*'*20)
    print(' '.join([f'{xx:10.3}' for xx in pbest]))
    print(' '.join([f'{xx:10.3}' for xx in pbest_e]))
    print('*'*20)
    return pbest, pbest_e, res[2]
