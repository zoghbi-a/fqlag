import numpy as np
from scipy.misc import derivative
import scipy.optimize as opt
import scipy.stats as st


def check_grad(mod, p0, dx=1e-3):
    """Compare the gradient from mod.loglikelihood_derivative
    against numerical derivative.
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


def maximize_old(mod, p0, limits=None):

    if limits is None:
        limits = [[-30,30] for x in p0]

    def f(x, mod):
        #x = np.clip(x, -30, 30)
        x = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)])
        try:
            l = mod.loglikelihood(x)
        except np.linalg.LinAlgError:
            l = -1e2
        return -l

    def fprime(x, mod):
        #x = np.clip(x, -30, 30)
        x = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)])
        try:
            l, g = mod.loglikelihood_derivative(x, calc_fisher=False)
        except np.linalg.LinAlgError:
            l = -1e6
            g = x*0 - 1e5
        print('%10.6g | %s | %s\r'%(l, 
                ' '.join(['%10.3g'%xx for xx in x]), ' '.join(['%10.3g'%xx for xx in g])), end="")
        return -g

    res = opt.minimize(f, p0, args=(mod), method='BFGS', tol=1e-4, jac=fprime, 
                options={'gtol':1e-4})
    print('\n** done **\n')
    p, pe = res.x, np.diag(res.hess_inv)**0.5
    return p, pe, res


def run_mcmc(mod, p0, perr=None, nwalkers=-10, nrun=100, **kwargs):

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
        

def maximize(mod, p0, limits=None, ipfix=None, verbose=1):

    if limits is None:
        limits = [[-30,30] for x in p0]

    if ipfix is None:
        ipfix = []
    npar = len(p0)
    pfix = np.array([p0[i] for i in ipfix])
    ivar = [i for i in range(npar) if not i in ipfix]
    info = [npar, pfix, ipfix, ivar]

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
            l = -1e2
        return -l

    def fprime(x, mod, info):
        npar, pfix, ipfix, ivar = info
        x = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(x,limits)])
        y = np.zeros(npar, np.double)
        y[ipfix] = pfix
        y[ivar ] = x
        #y = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(y,limits)])

        try:
            l, g = mod.loglikelihood_derivative(y, calc_fisher=False)
            g = g[ivar]
        except np.linalg.LinAlgError:
            l = -1e6
            g = x*0 - 1e5
        if verbose:
            #print('%10.6g | %s | %s\r'%(l, 
            #    ' '.join(['%10.3g'%xx for xx in x]), ' '.join(['%10.3g'%xx for xx in g])), end="")
            print('%10.6g | %s | %s\r'%(l, 
                ' '.join(['%10.3g'%xx for xx in x]), '%10.3g'%np.max(np.abs(g))), end="")
        return -g

    res = opt.minimize(f, p0[ivar], args=(mod, info), method='BFGS', tol=1e-4, jac=fprime, 
                options={'gtol':1e-4})
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


def maximize_no_grad(mod, p0, limits=None, ipfix=None, verbose=1):

    if limits is None:
        limits = [[-30,30] for x in p0]

    if ipfix is None:
        ipfix = []
    npar = len(p0)
    pfix = np.array([p0[i] for i in ipfix])
    ivar = [i for i in range(npar) if not i in ipfix]
    info = [npar, pfix, ipfix, ivar]

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
            l = -1e2
        
        print('%10.6g | %s \r'%(l, ' '.join(['%10.3g'%xx for xx in x])), end="")
        
        return -l


    res = opt.minimize(f, p0[ivar], args=(mod, info), method='BFGS', tol=1e-4, 
                options={'gtol':1e-4})
    if verbose: print('\n** done **\n')

    p, pe = res.x, np.diag(res.hess_inv)**0.5
    y, ye = np.zeros(npar, np.double), np.zeros(npar, np.double)
    y[ipfix] = pfix
    y[ivar ] = p
    ye[ivar] = pe 
    y = np.array([np.clip(xx, l[0], l[1]) for xx,l in zip(y,limits)])

    return y, ye, res


def step_par(mod, p0, par1, par2=None, **kwargs):
    """
    par1: [ip, p_array]
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
  
    

def errors(mod, p0, limits=None, ipars=None, **kwargs):

    tol     = kwargs.get('tol', 1e-2)
    sign    = kwargs.get('sign', 1)
    verbose = kwargs.get('verbose', True)
    DCHI2   = kwargs.get('DCHI2', 1.0)
    skip_ipars = kwargs.get('skip_ipars', [])

    npar = len(p0)
    if ipars is None:
        ipars = list(range(npar))
    ipars = [i for i in ipars]

    pbest, pbest_e, res = maximize(mod, np.array(p0)+1e-3, limits)
    lbest = -res.fun

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
                         'Try using sign=%d')%(ipar, sign, sign))
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

