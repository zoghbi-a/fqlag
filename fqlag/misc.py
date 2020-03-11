import numpy as np
from scipy.misc import derivative
import scipy.optimize as opt


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


def maximize(mod, p0):

    def f(x, mod):
        x = np.clip(x, -30, 30)
        try:
            l = mod.loglikelihood(x)
        except np.linalg.LinAlgError:
            l = -1e6
        return -l

    def fprime(x, mod):
        x = np.clip(x, -30, 30)
        try:
            l, g = mod.loglikelihood_derivative(x, calc_fisher=False)
        except np.linalg.LinAlgError:
            l = -1e6
            g = x*0 - 1e2
        print('%10.6g | %s | %s'%(l, 
                ' '.join(['%10.3g'%x for x in x]), ' '.join(['%10.3g'%x for x in g])))
        return -g

    res = opt.minimize(f, p0, args=(mod), method='BFGS', tol=1e-4, jac=fprime, 
                options={'gtol':1e-4})
    
    print(res)
    p, pe = res.x, np.diag(res.hess_inv)**0.5
    return p, pe


def optimize(mod, p0, tol=1e-4, maxiter=30, **kwargs):

    gzero      = kwargs.get('gzero', 1e-5)
    nsearch    = kwargs.get('nsearch', 30)
    rho        = kwargs.get('rho', 0.5)
    check_pars = kwargs.get('check_pars', None)
    check_dpar = kwargs.get('check_dpar', None) 
    ip_fix     = kwargs.get('ip_fix', None)
    use_bfgs   = kwargs.get('use_bfgs', False)
    wolf_c     = 1e-2


    pars   = np.array(p0)
    npar   = len(pars)
    dpar   = np.zeros(npar)
    saved  = [pars, -np.inf]


    #########
    def f(x, mod):
        try:
            l = mod.loglikelihood(x)
        except np.linalg.LinAlgError:
            l = -np.inf
        return -l
    def fprime(x, mod):
        try:
            l, g = mod.loglikelihood_derivative(x, calc_fisher=False)
        except np.linalg.LinAlgError:
            l = -np.inf
            g = x*0 - 1e2
        return -g


    func_calls, f = opt.optimize.wrap_function(f, (mod,))
    grad_calls, myfprime = opt.optimize.wrap_function(fprime, (mod,))

    N = len(p0)
    xk = p0
    pk = None
    gfk = myfprime(p0)
    old_fval = f(p0)
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    I = numpy.eye(N, dtype=int)
    Hk = I


    alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     opt.optimize._line_search_wolfe12(f, myfprime, xk, pk, gfk,old_fval, old_old_fval)


    from IPython import embed;embed();exit(0)


    from scipy.optimize import linesearch

    func_calls, f = opt.wrap_function(f, args)

    

    ##########



    for _iter in range(1, maxiter+1):

        # ----------------------------- #
        # check if we have a valid step #
        alpha = 1.0
        for isearch in range(nsearch):
            pars = saved[0] + alpha * dpar
            try:
                l = mod.loglikelihood(pars)
            except np.linalg.LinAlgError:
                if _iter == 1:
                    raise ValueError('input p0 does not produce a valid covariance!')
                if isearch == nsearch - 1:
                    raise ValueError('Cannot make progress in the search')
                alpha *= rho
                continue

            if _iter == 1: break

            wolf = wolf_c * alpha * np.dot(grad, dpar)
            if l - saved[1] < wolf:
                alpha *= rho
                print(alpha, isearch)
                continue
            break


        # ----------------------------- #
        # We have a valid step, take it #
        loglike, grad, fisher = mod.loglikelihood_derivative2(pars)
        gtol  = np.abs(grad).max()
        ftol  = np.abs(loglike - saved[1]) 
        print('%10d %10.4g %10.6g | %s | %s'%(_iter, alpha, loglike, 
                ' '.join(['%10.3g'%x for x in pars]), ' '.join(['%10.3g'%x for x in grad])))

        # ------------ #
        # are we done? #
        if gtol < tol:
            break



        # ----------------------------------------- #
        # prepare for next step, handling constants #
        ivar = np.argwhere(np.abs(grad)>gzero)[:,0]
        if len(ivar) == 0: ivar = np.arange(npar)
        if ip_fix is not None:
            if not isinstance(ip_fix, list): ip_fix = [ip_fix]
            ivar = [i for i in ivar if i not in ip_fix]
        try:
            ifisher = np.linalg.inv(fisher[np.ix_(ivar, ivar)])
        except:
            ifisher = np.linalg.inv(fisher[np.ix_(ivar, ivar)] + np.eye(len(ivar)))
        dpar   = np.zeros(npar)
        dpar[ivar] = np.dot(ifisher, grad[ivar])    



        # ----------------------- #
        # call any user functions #
        if not check_dpar is None: dpar = check_dpar(dpar)
        if not check_pars is None: pars = check_pars(pars)

        
        # ---------------------------------------- #
        # save parameter and move to the next step #
        saved = [pars, loglike, grad]
        

    p  = pars
    pe = np.diag(np.linalg.inv(fisher))**0.5

    
    print('\t***', '-'*37)
    print('\t***%15s: %10d'%('niter', _iter))
    print('\t***%15s: %10.4g'%('gtol', gtol))
    print('\t***%15s: %10.4g'%('ftol', ftol))
    print('\t***%15s: %10.6g'%('loglikelihood', loglike))
    print('\t***%15s: %s'%('parameters', ' '.join(['%10.3g'%x for x in p])))
    print('\t***%15s: %s'%('errors', ' '.join(['%10.3g'%x for x in pe])))


    if use_bfgs:

        def fun(x, mod):
            try:
                l, g = mod.loglikelihood_derivative(x, calc_fisher=False)
            except np.linalg.LinAlgError:
                l = -np.inf
                g = x*0 - 1e2
            print('%10.6g | %s | %s'%(l, 
                ' '.join(['%10.3g'%x for x in x]), ' '.join(['%10.3g'%x for x in g])))
            return -l, -g

        def hfun(x, mod):
            try:
                l, g,h = mod.loglikelihood_derivative(x)
            except np.linalg.LinAlgError:
                l = -np.inf
                g = x*0 - 1e2
                h = np.eye(len(x)) + 1
            print('hfun')
            return h

        #res = opt.minimize(fun, pars, args=(mod,), tol=tol, jac=True,
        #   method='BFGS', options={'gtol': gzero, 'disp':True})
        res = opt.minimize(fun, pars, args=(mod,), tol=tol, jac=True, hess=hfun,
           method='Newton-CG', options={'gtol': gzero, 'disp':True})
        
        p  = res.x
        l, g, h = mod.loglikelihood_derivative(res.x)
        pe = np.diag(np.linalg.inv(h))**0.5

        print('\n')
        print('\t***', '-'*37)
        print('\t***%15s: %10.4g'%('gtol', np.abs(res.jac).max()))
        print('\t***%15s: %s'%('parameters', ' '.join(['%10.3g'%x for x in p])))
        print('\t***%15s: %s'%('errors', ' '.join(['%10.3g'%x for x in pe])))
        print('\t***', '-'*37)
        print('\n')


    return p, pe


