
import numpy as np
import aztools as az
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import fqlag
from fqlag.misc import check_grad



def simulate_lc(n, dt, mu, lag, seed=3343, bkp=None, phase=True):
    np.random.seed(seed)
    sim = az.SimLC(32)
    if bkp is None: bkp = [1e-4, -1, -2, 5e-2]
    sim.add_model('broken_powerlaw', bkp)
    sim.add_model('constant', lag, lag=True)
    sim.simulate(n*3, dt, mu, 'rms')
    sim.apply_lag(phase=phase)
    tarr, rarr, sarr = sim.t[:n], sim.x[:n], sim.y[:n]
    #plt.plot(tarr, rarr, tarr, sarr)
    rarr = np.random.poisson(rarr)
    sarr = np.random.poisson(sarr)
    rerr = rarr*0 + np.mean(rarr**0.5)
    serr = sarr*0 + np.mean(sarr**0.5)
    
    #plt.errorbar(tarr, rarr, rerr, alpha=0.4)
    #lt.errorbar(tarr, sarr, serr, alpha=0.4)
    #plt.show()

    return tarr, rarr, rerr, sarr, serr, sim



def maximize(mod, p0):
    import scipy.optimize as opt

    def f(x, mod):
        try:
            l = mod.loglikelihood(x)
        except np.linalg.LinAlgError:
            l = -1e6
        return -l

    def fprime(x, mod):
        try:
            l, g = mod.loglikelihood_derivative(x, calc_fisher=False)
        except np.linalg.LinAlgError:
            l = -1e6
            g = x*0 - 1e2
        print('%10.6g | %s | %s'%(l, 
                ' '.join(['%10.3g'%x for x in x]), ' '.join(['%10.3g'%x for x in g])))
        return -g

    res = opt.minimize(f, p0, args=(mod), method='BFGS', tol=1e-4, jac=fprime,
        options={'gtol':1e-4}, bounds=[(-20,20)]*len(p0))
    
    print(res)
    p, pe = res.x, np.diag(res.hess_inv)**0.5
    #print(np.diag(res.hess_inv)**0.5)
    return p, pe


def test_base():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    import fqlag.base
    mod = fqlag.base.FqLagBase(tarr, rarr, rerr)
    p0 = [1e-2, 80.0]
    check_grad(mod, p0)



def test_psd():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.4), 4), [1.]])
    p0 = (fql[1:]**-1)

    pmod = fqlag.Psd(tarr, rarr, rerr, fql)
    check_grad(pmod, p0); return

    #from IPython import embed; embed();exit(0)

    maximize(pmod, p0)
    #p = optimize(pmod, p0, maxiter=30, use_bfgs=False)#[0]
    #from IPython import embed; embed();exit(0)
    return

    y = pmod.sample(p, 3)
    plt.plot(tarr, rarr-rarr.mean())
    for x in y:
        plt.plot(tarr, x, lw=0.5)
    plt.show()


def test_lpsd():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.4), 4), [1.]])
    p0 = np.log(fql[1:]**-2)

    #lpmod = fqlag.lPsd(tarr, rarr, rerr, fql, dt)
    lpmod = fqlag.Psd(tarr, rarr, rerr, fql, dt, log=True)
    #check_grad(lpmod, p0); return

    p = maximize(lpmod, p0)
    return


    y = lpmod.sample(p, 3)
    plt.plot(tarr, rarr-rarr.mean())
    for x in y:
        plt.plot(tarr, x, lw=0.5)
    plt.show()


def test_lpsd_sim():
    # works with bias and the factor of 2
    # or with no-bias and without the fator of 2
    nsim = 30
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    sim = az.SimLC(323)
    sim.add_model('broken_powerlaw', [1e-4, -1, -2, 1e-3])


    fql = np.concatenate([[0.1/n], np.logspace(np.log10(1.1/n), np.log10(0.4), 7)[1:], [2.]])
    p0 = fql[1:]*0 - 1
    #p0 = fql[1:]*0+0.1


    P = []
    for isim in range(1, nsim+1):
        sim.simulate(n, dt, mu, 'rms')
        tarr, rarr = sim.t[:n], sim.x[:n]
        rerr = rarr *0
        #rarr = np.random.poisson(rarr)
        #rerr = rarr*0 + np.mean(rarr**0.5)

        pmod = fqlag.lPsd(tarr, rarr, rerr, fql)
        #pmod = fqlag.Psd(tarr, rarr, rerr, fql)

        p = maximize(pmod, p0)
        P.append(p)
    P = np.array(P)
    #from IPython import embed; embed();exit(0)

    f,fp = sim.normalized_psd
    plt.semilogx(f[1:-1], np.log(fp[1:-1]))
    #plt.loglog(f[1:-1], fp[1:-1])

    fq = (fql[1:] + fql[:-1])/2
    plt.errorbar(fq, P.mean(0), P.std(0), fmt='o-')
    plt.show()


def test_psd_sim():
    # no bias needed. works without the fator of 2
    # so not using the factor of 2 is the correct way, and when doing the logs
    # no bias correction is needed
    nsim = 20
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    sim = az.SimLC(323)
    sim.add_model('broken_powerlaw', [1e-4, -1, -2, 1e-3])


    fql = np.concatenate([[0.1/n], np.logspace(np.log10(1.1/n), np.log10(0.4), 7)[1:], [2.]])
    p0 = fql[1:]*0+0.1


    P = []
    for isim in range(1, nsim+1):
        sim.simulate(n, dt, mu, 'rms')
        tarr, rarr = sim.t[:n], sim.x[:n]
        rerr = rarr *0
        #rarr = np.random.poisson(rarr)
        #rerr = rarr*0 + np.mean(rarr**0.5)

        #pmod = fqlag.lPsd(tarr, rarr, rerr, fql)
        pmod = fqlag.Psd(tarr, rarr, rerr, fql)

        p = maximize(pmod, p0)
        P.append(p)
    P = np.array(P)
    #from IPython import embed; embed();exit(0)

    f,fp = sim.normalized_psd
    #plt.semilogx(f[1:-1], np.log(fp[1:-1]))
    plt.loglog(f[1:-1], fp[1:-1])

    fq = (fql[1:] + fql[:-1])/2
    plt.errorbar(fq, P.mean(0), P.std(0), fmt='o-')
    plt.show()

    
def test_psdf():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag, bkp=[1e-4, -1, -2, 5e-2])

    fql = np.array([.1/n, 2.0])
    p0 = np.array([-5., -2.])
    

    pmod = fqlag.Psdf(tarr, rarr, rerr, fql, 'pl', dt=None)
    #print(pmod.loglikelihood(p0))
    #check_grad(pmod, p0); return

    p = maximize(pmod, p0)
    print(p[0], p[1])
    return


    y = pmod.sample(p[0], 3)
    plt.plot(tarr, rarr-rarr.mean())
    for x in y:
        plt.plot(tarr, x, lw=0.5)
    plt.show()


def test_psdf_sim__pl():
    # works with bias and the factor of 2
    # or with no-bias and without the fator of 2
    nsim = 100
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    sim = az.SimLC(323)
    sim.add_model('powerlaw', [1e-4, -2])
    #sim.add_model('bending_powerlaw', [1e-4, -2, 5e-2])


    fql = np.array([0.5/n, 1])
    p0 = np.array([-9., -2.])
    p0 = np.array([-9., -2., -4.5])


    P = []
    for isim in range(1, nsim+1):
        sim.simulate(n, dt, mu, 'rms')
        tarr, rarr = sim.t[:n], sim.x[:n]
        rerr = rarr *0
        #rarr = np.random.poisson(rarr)
        #rerr = rarr*0 + np.mean(rarr**0.5)

        pmod = fqlag.Psdf(tarr, rarr, rerr, fql, 'bpl', dt=None)

        p = maximize(pmod, p0)
        P.append(p)
    P = np.array(P)
    #from IPython import embed; embed();exit(0)

    f,fp = sim.normalized_psd
    #plt.semilogx(f[1:-1], np.log(fp[1:-1]))
    plt.loglog(f[1:-1], fp[1:-1])

    fq = np.logspace(np.log10(fql[0]), np.log10(fql[-1]), 40)
    pp = np.array([pmod.psd_func(fq, x[0]) for x in P])
    pm,ps = pp.mean(0), pp.std(0)
    plt.fill_between(fq, pm-ps, pm+ps, alpha=0.5)
    plt.show()


def test_cxd():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    fql = np.concatenate([[0.1/n], np.logspace(np.log10(1.1/n), np.log10(0.4), 7)[1:], [2.]])
    #fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
    p0 = fql[1:]*0 + 0.1

    #p1mod = fqlag.Psd(tarr, rarr, rerr, fql, dt)
    #p1 = optimize(p1mod, p0)[0]
    #p1 = maximize(p1mod, p0)
    p1 = [np.array([0.13035129, 0.19516932, 0.06885694, 0.03005373, 0.0057864 ,
       0.00104531, 0.00060654])]


    #p2mod = fqlag.Psd(tarr, sarr, serr, fql, dt)
    #p2 = optimize(p2mod, p1)[0]
    #p2 = maximize(p2mod, p1[0])
    p2 = [np.array([0.04900969, 0.12973469, 0.08798995, 0.03375997, 0.00607322,
       0.00091455, 0.00059197])]


    #p0 = np.concatenate([np.min([p1[0],p2[0]], 0)*0.7, p1[0]*0+1])
    #cmod = fqlag.Cxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1[0], p2[0], dt)
    
    p0 = np.concatenate([p1[0], p2[0], np.min([p1[0],p2[0]], 0)*0.7, p1[0]*0+1])
    cmod = fqlag.PCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, dt)
    check_grad(cmod, p0, dx=1e-5); return


    print('-----------------')

    #c = optimize(cmod, p0, maxiter=10, use_bfgs=True, check_dpar=lambda dp:np.clip(dp, -10, 10))
    c = maximize(cmod, p0)



def test_lcxd():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    
    #fql = np.concatenate([[0.5/n], np.logspace(np.log10(1.5/n), np.log10(0.4), 6)[1:], [1.]])
    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 4)])
    p0 = fql[1:]*0 - 1

    #p1mod = fqlag.lPsd(tarr, rarr, rerr, fql, )
    p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
    #p1 = optimize(p1mod, p0, check_dpar=lambda dp:np.clip(dp, -4, 4))[0]
    p1 = maximize(p1mod, p0)
    #p1 = [np.array([-1.98476999, -2.10760789, -3.85694557, -6.82487901])]
    

    #p2mod = fqlag.lPsd(tarr[1:], sarr[1:], serr[1:], fql, )
    p2mod = fqlag.Psd(tarr[1:], sarr[1:], serr[1:], fql, log=True)
    #p2 = optimize(p2mod, p1, check_dpar=lambda dp:np.clip(dp, -4, 4))[0]
    p2 = maximize(p2mod, p1[0])
    #p2 = [np.array([-2.84762631, -2.16740838, -3.75367259, -6.8628747 ])]


    #p0 = np.concatenate([np.min([p1[0],p2[0]], 0)-3, p1[0]*0+0.5])
    #cmod = fqlag.lCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1[0], p2[0], )
    #cmod = fqlag.Cxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1[0], p2[0], log=True)

    p0 = np.concatenate([p1[0], p2[0], np.min([p1[0],p2[0]], 0)-3, p1[0]*0+1])
    #cmod = fqlag.lPCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, dt)
    cmod = fqlag.PCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, dt, log=True)
    check_grad(cmod, p0, 1e-5); return
    #print(cmod.loglikelihood(p0));return

    print('-----------------\n')
    
    
    #c = optimize(cmod, p0, maxiter=10, use_bfgs=True, check_dpar=lambda dp:np.clip(dp, -4, 4))
    c = maximize(cmod, p0)


def test_psi():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
    p0 = fql[1:]**-2

    p1mod = fqlag.Psd(tarr, rarr, rerr, fql)
    p1 = maximize(p1mod, p0)[0]


    cmod = fqlag.Psi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1)
    p0 = np.concatenate([p1*0.1, p1*0])
    fqlag.misc.check_grad(cmod, p0)
    
    #c = maximize(cmod, p0)



def test_lpsi():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
    p0 = np.log(fql[1:]**-2)

    #p1mod = fqlag.lPsd(tarr, rarr, rerr, fql)
    p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
    p1 = maximize(p1mod, p0)[0]


    #cmod = fqlag.lPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1)
    cmod = fqlag.Psi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1, log=True)
    p0 = np.concatenate([p1*0-0.5, p1*0])
    fqlag.misc.check_grad(cmod, p0); return
    
    c = maximize(cmod, p0)


def test_ppsi():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 4)])
    p0 = fql[1:]**-2

    p1mod = fqlag.Psd(tarr, rarr, rerr, fql)
    #p1 = optimize(p1mod, p0)[0]
    p1 = maximize(p1mod, p0)[0]

    cmod = fqlag.PPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql)
    p0 = np.concatenate([p1, p1*0+1.0, p1*0])
    fqlag.misc.check_grad(cmod, p0, 1e-4); return

    c = maximize(cmod, p0)


def test_lppsi():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
    p0 = np.log(fql[1:]**-2)

    #p1mod = fqlag.lPsd(tarr, rarr, rerr, fql)
    p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
    p1 = maximize(p1mod, p0)[0]


    #cmod = fqlag.lPPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql)
    cmod = fqlag.PPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, log=True)
    p0 = np.concatenate([p1, p1*0-0.5, p1*0])
    fqlag.misc.check_grad(cmod, p0, 1e-6); return
    c = maximize(cmod, p0)


def test_cxdRI():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    #fql = np.concatenate([[0.8/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [0.6]])
    fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 4)])
    p0 = (fql[1:]**-2)

    p1mod = fqlag.Psd(tarr, rarr, rerr, fql)
    #p1 = optimize(p1mod, p0)[0]
    p1 = maximize(p1mod, p0)

    p2mod = fqlag.Psd(tarr, sarr, serr, fql)
    #p2 = optimize(p2mod, p1)[0]
    p2 = maximize(p2mod, p1)

    p0 = np.concatenate([(p1+p2)*0.3, (p1+p2)*0.3])
    cmod = fqlag.CxdRI([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1, p2)

    fqlag.misc.check_grad(cmod, p0);return
    #c = optimize(cmod, p0, tol=1e-3, maxiter=20, use_bfgs=True)
    c = maximize(cmod, p0)
    

def test_psif():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)
    
    fql = np.array([.1/n, 2.0])
    p0 = np.array([-5., -2.])
    

    pmod = fqlag.Psdf(tarr, rarr, rerr, fql, 'pl', dt=None)
    #print(pmod.loglikelihood(p0))
    #check_grad(pmod, p0); return

    p = maximize(pmod, p0)
    #return


    cmod = fqlag.Psif([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p[0], ['pl', 'expc', 'c'])
    p0 = np.array([0., 0.])
    #print(cmod.loglikelihood(p0)); return
    #fqlag.misc.check_grad(cmod, p0); return
    
    c = maximize(cmod, p0)
    print(c[0], c[1])
    print()
    
    # PPsif
    cmod = fqlag.PPsif([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, ['pl', 'expc', 'c'])
    p0 = np.concatenate([p[0], c[0]])
    p0 = np.concatenate([p[0], [-0.8, 1]])
    #print(cmod.loglikelihood(p0)); return
    #fqlag.misc.check_grad(cmod, p0); return

    c = maximize(cmod, p0)
    print(c[0], c[1])
    


def test_storm():

    base = ['1158', '1367', '1479', '1746', 'B', 'g', 'I1', 'i', 
            'R1', 'r', 'u', 'V', 'z']
    
    ff = []
    for b in base:
        tt = np.loadtxt('storm5548/%s.dat'%(b))[:,0]
        tt = np.unique(np.abs(tt - tt[:,None]))
        tt = tt[tt > 1]
        ff.append([1./tt[-1], 1./tt[0]])
    ff = np.array(ff)
    f1 = ff[:,0].max()
    f2 = ff[:,1].min()
    f0 = ff[:,0].min()
    fe = ff[:,1].max()

    #fqL = np.logspace(np.log10(0.9*f1), np.log10(0.1*f2), 5)
    fqL = np.concatenate([[0.5*f0], np.logspace(np.log10(2*f1), np.log10(0.1*f2), 5), [2*fe]])
    #fqL = np.array([1e-4, 0.008, 0.02, 0.1, 100])


    nfq = len(fqL) - 1
    fqd = (fqL[1:] + fqL[:-1])/2





    iref = 0
    ilc  = 7

    tr, lr, lre = np.loadtxt('storm5548/%s.dat'%base[iref]).T
    t , l , le  = np.loadtxt('storm5548/%s.dat'%base[ilc]).T



    p1mod = fqlag.lPsd(tr, lr, lre, fqL)
    p0 = np.zeros(nfq) - 1
    p1 = maximize(p1mod, p0)[0]

    ## --
    cmod = fqlag.lPsi([tr, t], [lr, l], [lre, le], fqL)
    p0 = np.concatenate([p1, p1*0, p1*0])
    c = maximize(cmod, p0)
    ## ---

    
    from IPython import embed; embed();exit(0)



    # shape: len(base)-1, 2, nfq*3
    C = np.array(C)
    psd, amp, phi = np.split(C, 3, axis=-1)
    psd, amp, phi = psd[:,:,1:-1], amp[:,:,1:-1], phi[:,:,1:-1]
    phi = (phi + np.pi) % (2*np.pi) - np.pi
    fq = fqd[1:-1]
    lag = phi / (2*np.pi*fq)
    xax = np.arange(len(psd)) + 1


    txt = 'descriptor number band %s'%(' '.join(['amp_f%d,+- phase_f%d,+- tau_f%d,+-'%(i+1,i+1,i+1)
                                                for i in range(len(fq))]))

    txt += '\n'.join(['%d "%s" %s'%(ii, base[ii], ' '.join(['%g %g %g %g %g %g'%(
            amp[ii,0,ifq], amp[ii,1,ifq], phi[ii,0,ifq], phi[ii,1,ifq], lag[ii,0,ifq], lag[ii,1,ifq]) 
            for ifq in range(len(fq))])) for ii in range(len(psd))])
    with open('storm_lags.plot', 'w') as fp: fp.write(txt)



    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    for ii in range(len(fq)):
        ax[0].errorbar(xax, psd[:,0,ii], psd[:,1,ii], fmt='o-')
        ax[1].errorbar(xax, amp[:,0,ii], amp[:,1,ii], fmt='o-')
        ax[2].errorbar(xax, phi[:,0,ii], phi[:,1,ii], fmt='o-')
        ax[3].errorbar(xax, lag[:,0,ii], lag[:,1,ii], fmt='o-')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    #test_base()
    #test_psd()
    #test_lpsd()
    #test_lpsd_sim()
    #test_psd_sim()
    #test_psdf()
    #test_psdf_sim__pl()
    
    #test_cxd()
    #test_lcxd()
    #test_psi()
    #test_lpsi()
    #test_ppsi()
    test_lppsi()
    #test_psif()
    
    #test_cxdRI()

    #test_storm()


