import numpy as np
from scipy.misc import derivative
import scipy.optimize as opt
import scipy.stats as st


def check_grad(mod, p0, dx=1e-3):
    """Compare the gradient from mod.loglikelihood_derivative
    against numerical derivative.

    Tests that the derivatie codes are correct

    Args:
        mod: the model we are testing
        p0: parameters of the model
        dx: used for the numerical derivative

    Returns:
        logLikelihood, grad_array_analytic, grad_array_numerical
    """

    p0 = np.array(p0)
    l,g,h = mod.loglikelihood_derivative(p0)
    
    g0 = []
    for i in range(len(p0)):
        def f(x, p0):
            p0[i] = x
            return mod.loglikelihood(p0)
        g0.append(derivative(f, p0[i], dx, args=(np.array(p0), )))
    g0 = np.array(g0)
    
    print('analytic:  ', ' '.join(['%10.4g'%x for x in g]))
    print('numerical: ', ' '.join(['%10.4g'%x for x in g0]))
    return l, g, g0


def run_mcmc(mod, p0, perr=None, nwalkers=-10, nrun=100, **kwargs):
    """Run MCMC to estimate the Posterior distributions of the parameters

    This uses the emcee package.

    Args:
        mod: the model object for calulating the likelihood function
        p0: starting paramters for chain
        perr: estimated uncertainties on the parameters. They are used to 
            initialize the chains. If not given, the chain is started with 
            values within 10% of p0
        nwalkers: number of walkers in the chains (see emcee for details)
        nrun: number of chain runs

    Keywords:
        sigma_f: the factor that multiples perr used to initialize the 
            walkers. Default: 0.5
        limits: a list of [pmin, pmax] values for the limits on the parameters.
            These are effectively used as uniform priors on the parameters
        iphi: The indicies of the phase parameters within p0. Used to ensure
            that those parameters are cyclic and remain -pi < phi < pi

    Returns:
        the chain array where the walker axis is flattened.

    """

    sigma_f = kwargs.get('sigma_f', 0.5)
    limits  = kwargs.get('limits', None)
    iphi    = kwargs.get('iphi', None)
    if limits is None:
        limits = [[-30,30] for x in p0]
    if iphi is None:
        iphi = []
    
    def logProb(x, mod):
        for ix in range(len(x)):
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
    
    try:
        import emcee
    except ModuleNotFoundError:
        raise RuntimeError('Cannot find emcee. Please install it first')
    
    ndim = len(p0)
    if nwalkers < 0: nwalkers = -ndim * nwalkers
    pe = p0 * 0.1 if perr is None else perr
        
    p0      = np.random.randn(nwalkers, ndim)*pe*sigma_f + p0
    p0 = np.array([[np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)] for x in p0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb, args=[mod,])
    state   = sampler.run_mcmc(p0, nrun)
    pchain  = sampler.flatchain
    lchain  = sampler.flatlnprobability
    print('acceptance fraction: ', np.mean(sampler.acceptance_fraction))
    chain   = np.hstack([pchain, np.expand_dims(lchain, -1)])
    return chain
        

def maximize(mod, p0, limits=None, ipfix=None, verbose=1, useGrad=False):
    """Maixmize the likelihood of model mod
    
    Use numerical optimization from scipy.optimize.minimize to estimate
    the parameters of the model at the likelihood maximum
    We the BFGS algorithm.

    Args:
        mod: model whose likelihood is to be optimized
        p0: starting model parameters
        limits: a list of [pmin, pmax] values for the limits on the parameters.
            These are effectively used as uniform priors on the parameters.
            None means all parameters are assumed to be between [-30, 30]
        ipfix: parameter indices of p0 to keep fixed during the maximization.
            Useful when calculating uncertainties by stepping through them.
        verbose: if True, print progress
        useGrad: use analytical gradient. This may give ~10% speedup, but it can be
            unstable for complex problems.
    
    Returns: 
        return (pars_best, pars_best_error, fit_result) 
            the latter is from scipy.optimize.minimize

    """

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
            print('%10.6g | %s \r'%(l, ' '.join(['%10.3g'%xx for xx in x])), end="")

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
            #print('%10.6g | %s | %s\r'%(l, 
            #    ' '.join(['%10.3g'%xx for xx in x]), ' '.join(['%10.3g'%xx for xx in g])), end="")
            print('%10.6g | %s | %s\r'%(l, 
                ' '.join(['%10.3g'%xx for xx in x]), '%10.3g'%np.max(np.abs(g))), end="")
        return -g

    if not useGrad:
        fprime = None
    res = opt.minimize(f, p0[ivar], args=(mod, info), method='BFGS', tol=1e-4, jac=fprime, 
                options={'gtol':1e-4})

    # last print 
    if verbose: 
        print('%10.6g | %s | %s\r'%(-res.fun, 
                ' '.join(['%10.3g'%xx for xx in res.x]), '%10.3g'%np.max(np.abs(res.jac))), end="")
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

    Args:
        mod: model whose likelihood is to be optimized
        p0: starting model parameters
        par1: [ipar, p_array], where ipar is the index of the parameter
            in p0 to step through, and p_array is the array of parameters
            to use
        par1: similar to par1 to do two parameters. Default is None, so we 
            only do one parameter

    Keywords;
        verbose: if True, print progress
        limits: a list of [pmin, pmax] values for the limits on the parameters.
            to be passed to @maximize
    
    Returns: 
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
    if verbose: print('best loglikelihood: %10.6g'%lbest)
    
    ip1 = par1[0] 
    step = []
    for iip1,p1 in enumerate(par1[1]):
        p = np.array(pbest)
        p[ip1] = p1
        if not par2 is None:
            ip2 = par2[0] 
            step2 = []
            for iip2,p2 in enumerate(par2[1]):
                pp = np.array(p)
                pp[ip2] = p2
                res = maximize(mod, pp, limits, ipfix=[ip1, ip2], verbose=False)
                step2.append([p1, p2, -res[2].fun])
                if verbose: 
                    print('%10.3g %10.3g %10.6g %10.3g\r'%(tuple(step2[-1])+(np.round(lbest - step2[-1][-1], 2),)), end="")
            step.append(step2)
        else:
            res = maximize(mod, p, limits, ipfix=[ip1], verbose=False)
            step.append([p1, -res[2].fun])
            if verbose: 
                print('%10.3g %10.6g %10.3g\r'%(tuple(step[-1])+(np.round(lbest - step[-1][-1], 2),)), end="")
    step = np.array(step)
    return step, [pbest, pbest_e, lbest]
  
    
def errors(mod, p0, ipars=None, **kwargs):
    """Calculate the uncertainties in the parameters p0 that maximize
    the likelihood function of a model mod.

    For each parameter, the value is changed in small steps until the log-likelihood
    changes by DCHI2 (default is 1, to calculated the 1-sigma uncertainties)

    Args:
        mod: model whose log-likelihood can called as mod.loglikelihood
        p0: the model parameters that maximize the likelihood, obtained 
            for example by running @misc.maximize
        ipars: a list of indices of p0 for which the errors are to be calculated
            Default: None, means calculate errors for all parameters


    Keywords:
        limits: a list of [pmin, pmax] values for the limits on the parameters.
            to be passed to @maximize
        tol: tolerance in loglikelihood value. e.g. calculation stop when
            |Delta(loglikelihood) - DCHI2| < tol. Default: 1e-2
        DCHI2: the change in loglikelihood value to probe. DCHI2=1 gives ~1-sigma
            uncertainties. For 90% confidence for instance, use DCHI2=2.71.
        skip_ipars: Parameter indices to skip in calculating the errors. Usefull
            for example in combination with ipars=None above.
        sign: the direction of the parameter uncertainty search. Default:1 means
            increase the parameter until Delta(loglikelihood)=DCHI2. -1 means seach
            in the other direction. Doing one direction assumes gaussian uncertainties.
            If the assumption breaks, both both +1 and -1 uncertainties should be reported.
        verbose: True to print progress


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

    ipars = [i for i in ipars]

    # make sure we start the search from parameters that maximize the likelihood
    pbest, pbest_e, res = maximize(mod, np.array(p0)+1e-3, limits)
    lbest = -res.fun

    # loop through the parameters. If a new maximum is found, 
    # restart the whole search
    p, pe = np.array(pbest), np.array(pbest_e)
    for iipar, ipar in enumerate(ipars):
        
        if iipar in skip_ipars:
            continue
        if verbose: print('\t## errors for param %d ##'%ipar)

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
                    print('%10.6g | %s \r'%(-tmp_res[2].fun, 
                        ' '.join(['%10.3g'%xx for xx in tmp_res[0]])), end="")
                #ipars = ipars[iipar:] + ipars[:iipar]
                return errors(mod, tmp_res[0], limits, ipars, **kwargs)
            isig += 0.5
            if isig >= 10:
                Warning(('parameter %d appears to be unbound using sign=%d\n'
                         'Try using sign=%d')%(ipar, sign, -sign))
                pbest_e[ipar] = np.abs(pExtrm - pbest[ipar])
                not_bound = True
                break
        if not_bound: continue
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
                    print('%10.6g | %s \r'%(-tmp_res[2].fun, 
                        ' '.join(['%10.3g'%xx for xx in tmp_res[0]])), end="")
                #ipars = np.concatenate([ipars[iipar:], ipars[:iipar]])
                return errors(mod, tmp_res[0], limits, ipars, **kwargs)
            if verbose:
                print(' %10.6g %10.6g %10.6g %10.6g %10.6g\r'%(
                    lbest, -tmp_res[2].fun, p[ipar], pHalf, dchi2), end="")
            if dchi2 < DCHI2: 
                pExtrm2 = pHalf
            else: 
                pExtrm = pHalf
            if icount >= 50: break
        pbest_e[ipar] = np.abs(pHalf - pbest[ipar])
        print()

    # get fit result again, so the return is consistent with the return of maximize
    res = maximize(mod, pbest, limits, verbose=False)
    
    # finally, if DCHI2 != 1; scale the errors, so the return always corresponds to 1-sigma errors
    error_scale = (st.norm.ppf((1-st.chi2.cdf(1, 1))/2) / st.norm.ppf((1-st.chi2.cdf(DCHI2, 1))/2))
    pbest_e[ipars] = pbest_e[ipars] * error_scale

    print('*'*20)
    print(' '.join(['%10.3g'%xx for xx in pbest]))
    print(' '.join(['%10.3g'%xx for xx in pbest_e]))
    print('*'*20)
    return pbest, pbest_e, res[2]

