"""Do basic runs to check that the functions run"""
import argparse as ARG

import numpy as np
import aztools as az
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))
import fqlag
from fqlag.misc import check_grad, maximize



def simulate_lc(n, dt, mu, lag, seed=3343, bkp=None, phase=True):
    """Simulate a light curve"""
    np.random.seed(seed)
    sim = az.SimLC(32)
    if bkp is None: bkp = [1e-3, -1, -2, 1e-3]
    sim.add_model('broken_powerlaw', bkp)
    sim.add_model('constant', [lag], lag=True)
    sim.simulate(n*3, dt, mu, norm='rms')
    sim.apply_lag(phase=phase)
    tarr, rarr, sarr = sim.lcurve[0][:n], sim.lcurve[1][:n], sim.lcurve[2][:n]
    #plt.plot(tarr, rarr, tarr, sarr)
    rarr = np.random.poisson(rarr)
    sarr = np.random.poisson(sarr)
    rerr = rarr*0 + np.mean(rarr**0.5)
    serr = sarr*0 + np.mean(sarr**0.5)

    return tarr, rarr, rerr, sarr, serr, sim


def _print(txt):
    print(f'\n{"-"*20}\n{txt}\n{"-"*20}\n')


def test_base():

    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    import fqlag.base
    mod = fqlag.base.FqLagBase(tarr, rarr, rerr)
    p0 = [1e-2, 80.0]
    _print('checking the gradient')
    check_grad(mod, p0)


def test_psd_nolog():

    _print('Running psd_nolog')
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    fql = np.logspace(np.log10(.5/n), np.log10(1.), 6)
    p0 = fql[1:]*0 + 0.1

    pmod = fqlag.Psd(tarr, rarr, rerr, fql, log=False)
    _print('checking the gradient')
    check_grad(pmod, p0)

    _print('Maximum likelihood')
    pfit = maximize(pmod, p0)
    p = pfit[0]

    _print('Sampling')
    y = pmod.sample(p, 3)
    plt.plot(tarr, rarr-rarr.mean())
    for x in y:
        plt.plot(tarr, x, lw=0.5)
    plt.savefig('figures/base_run__psd_nolog.png')


def test_psd_log():

    _print('Running psd_log')
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    fql = np.logspace(np.log10(.5/n), np.log10(1.), 6)
    p0 = fql[1:]*0 + 0.1

    pmod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
    _print('checking the gradient')
    check_grad(pmod, p0)

    _print('Maximum likelihood')
    pfit = maximize(pmod, p0)
    p = pfit[0]

    _print('Sampling')
    y = pmod.sample(p, 3)
    plt.plot(tarr, rarr-rarr.mean())
    for x in y:
        plt.plot(tarr, x, lw=0.5)
    plt.savefig('figures/base_run__psd_log.png')

    
def test_cxd_nolog():

    _print('Running cxd_nolog')
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)
    log = False

    fql = np.logspace(np.log10(.5/n), np.log10(1.), 6)
    p0 = fql[1:]*0 + 0.1

    _print('first psd')
    p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=log)
    p1fit = maximize(p1mod, p0)
    p1 = p1fit[0]
    
    _print('second psd')
    p2mod = fqlag.Psd(tarr, sarr, serr, fql, log=log)
    p2fit = maximize(p2mod, p0)
    p2 = p2fit[0]

    
    p0 = np.concatenate([np.min([p1, p2], 0)*0.7, p1*0+0.1])
    cmod = fqlag.Cxd([tarr, tarr], [rarr, sarr], [rerr, serr], fql, p1, p2, log=log)
    
    _print('checking the gradient')
    check_grad(cmod, p0)


    _print('Maximum likelihood')
    cfit = maximize(cmod, p0)


def test_cxd_log():

    _print('Running cxd_nolog')
    n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
    tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)
    log = True

    fql = np.logspace(np.log10(.5/n), np.log10(1.), 6)
    p0 = fql[1:]*0 + 0.1

    _print('first psd')
    p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=log)
    p1fit = maximize(p1mod, p0)
    p1 = p1fit[0]
    
    _print('second psd')
    p2mod = fqlag.Psd(tarr, sarr, serr, fql, log=log)
    p2fit = maximize(p2mod, p0)
    p2 = p2fit[0]

    
    p0 = np.concatenate([np.min([p1, p2], 0)-0.3, p1*0+0.1])
    cmod = fqlag.Cxd([tarr, tarr], [rarr, sarr], [rerr, serr], fql, p1, p2, log=log)
    
    _print('checking the gradient')
    check_grad(cmod, p0)


    _print('Maximum likelihood')
    cfit = maximize(cmod, p0)

# def test_lpsd():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.4), 4), [1.]])
#     p0 = np.log(fql[1:]**-2)

#     #lpmod = fqlag.lPsd(tarr, rarr, rerr, fql, dt)
#     lpmod = fqlag.Psd(tarr, rarr, rerr, fql, dt, log=True)
#     #check_grad(lpmod, p0); return

#     p = maximize(lpmod, p0)
#     return


#     y = lpmod.sample(p, 3)
#     plt.plot(tarr, rarr-rarr.mean())
#     for x in y:
#         plt.plot(tarr, x, lw=0.5)
#     plt.show()

    
# def test_psdf():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag, bkp=[1e-4, -1, -2, 5e-2])

#     fql = np.array([.1/n, 2.0])
#     p0 = np.array([-5., -2.])
    

#     pmod = fqlag.Psdf(tarr, rarr, rerr, fql, 'pl', dt=None)
#     #print(pmod.loglikelihood(p0))
#     #check_grad(pmod, p0); return

#     p = maximize(pmod, p0)
#     print(p[0], p[1])
#     return


#     y = pmod.sample(p[0], 3)
#     plt.plot(tarr, rarr-rarr.mean())
#     for x in y:
#         plt.plot(tarr, x, lw=0.5)
#     plt.show()


# def test_lcxd():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

    
#     #fql = np.concatenate([[0.5/n], np.logspace(np.log10(1.5/n), np.log10(0.4), 6)[1:], [1.]])
#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 4)])
#     p0 = fql[1:]*0 - 1

#     #p1mod = fqlag.lPsd(tarr, rarr, rerr, fql, )
#     p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
#     #p1 = optimize(p1mod, p0, check_dpar=lambda dp:np.clip(dp, -4, 4))[0]
#     p1 = maximize(p1mod, p0)
#     #p1 = [np.array([-1.98476999, -2.10760789, -3.85694557, -6.82487901])]
    

#     #p2mod = fqlag.lPsd(tarr[1:], sarr[1:], serr[1:], fql, )
#     p2mod = fqlag.Psd(tarr[1:], sarr[1:], serr[1:], fql, log=True)
#     #p2 = optimize(p2mod, p1, check_dpar=lambda dp:np.clip(dp, -4, 4))[0]
#     p2 = maximize(p2mod, p1[0])
#     #p2 = [np.array([-2.84762631, -2.16740838, -3.75367259, -6.8628747 ])]


#     #p0 = np.concatenate([np.min([p1[0],p2[0]], 0)-3, p1[0]*0+0.5])
#     #cmod = fqlag.lCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1[0], p2[0], )
#     #cmod = fqlag.Cxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1[0], p2[0], log=True)

#     p0 = np.concatenate([p1[0], p2[0], np.min([p1[0],p2[0]], 0)-3, p1[0]*0+1])
#     #cmod = fqlag.lPCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, dt)
#     cmod = fqlag.PCxd([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, dt, log=True)
#     check_grad(cmod, p0, 1e-5); return
#     #print(cmod.loglikelihood(p0));return

#     print('-----------------\n')
    
    
#     #c = optimize(cmod, p0, maxiter=10, use_bfgs=True, check_dpar=lambda dp:np.clip(dp, -4, 4))
#     c = maximize(cmod, p0)


# def test_psi():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

#     #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
#     p0 = fql[1:]**-2

#     p1mod = fqlag.Psd(tarr, rarr, rerr, fql)
#     p1 = maximize(p1mod, p0)[0]


#     cmod = fqlag.Psi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1)
#     p0 = np.concatenate([p1*0.1, p1*0])
#     fqlag.misc.check_grad(cmod, p0)


# def test_lpsi():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

#     #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
#     p0 = np.log(fql[1:]**-2)

#     #p1mod = fqlag.lPsd(tarr, rarr, rerr, fql)
#     p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
#     p1 = maximize(p1mod, p0)[0]


#     #cmod = fqlag.lPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1)
#     cmod = fqlag.Psi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1, log=True)
#     p0 = np.concatenate([p1*0-0.5, p1*0])
#     fqlag.misc.check_grad(cmod, p0); return
    
#     c = maximize(cmod, p0)


# def test_ppsi():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

#     #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 4)])
#     p0 = fql[1:]**-2

#     p1mod = fqlag.Psd(tarr, rarr, rerr, fql)
#     #p1 = optimize(p1mod, p0)[0]
#     p1 = maximize(p1mod, p0)[0]

#     cmod = fqlag.PPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql)
#     p0 = np.concatenate([p1, p1*0+1.0, p1*0])
#     fqlag.misc.check_grad(cmod, p0, 1e-4); return

#     c = maximize(cmod, p0)


# def test_lppsi():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

#     #fql = np.concatenate([[0.5/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [1.0]])
#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 6)])
#     p0 = np.log(fql[1:]**-2)

#     #p1mod = fqlag.lPsd(tarr, rarr, rerr, fql)
#     p1mod = fqlag.Psd(tarr, rarr, rerr, fql, log=True)
#     p1 = maximize(p1mod, p0)[0]


#     #cmod = fqlag.lPPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql)
#     cmod = fqlag.PPsi([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, log=True)
#     p0 = np.concatenate([p1, p1*0-0.5, p1*0])
#     fqlag.misc.check_grad(cmod, p0, 1e-6); return
#     c = maximize(cmod, p0)


# def test_cxdRI():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)

#     #fql = np.concatenate([[0.8/n], np.logspace(np.log10(2/n), np.log10(0.4), 6), [0.6]])
#     fql = np.concatenate([[0.1/n], np.logspace(np.log10(2/n), np.log10(0.6), 4)])
#     p0 = (fql[1:]**-2)

#     p1mod = fqlag.Psd(tarr, rarr, rerr, fql)
#     #p1 = optimize(p1mod, p0)[0]
#     p1 = maximize(p1mod, p0)

#     p2mod = fqlag.Psd(tarr, sarr, serr, fql)
#     #p2 = optimize(p2mod, p1)[0]
#     p2 = maximize(p2mod, p1)

#     p0 = np.concatenate([(p1+p2)*0.3, (p1+p2)*0.3])
#     cmod = fqlag.CxdRI([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p1, p2)

#     fqlag.misc.check_grad(cmod, p0);return
#     #c = optimize(cmod, p0, tol=1e-3, maxiter=20, use_bfgs=True)
#     c = maximize(cmod, p0)
    

# def test_psif():

#     n, dt, mu, lag = 2**8, 1.0, 10000, 1.0
#     tarr, rarr, rerr, sarr, serr, sim = simulate_lc(n, dt, mu, lag)
    
#     fql = np.array([.1/n, 2.0])
#     p0 = np.array([-5., -2.])
    

#     pmod = fqlag.Psdf(tarr, rarr, rerr, fql, 'pl', dt=None)
#     #print(pmod.loglikelihood(p0))
#     #check_grad(pmod, p0); return

#     p = maximize(pmod, p0)
#     #return


#     cmod = fqlag.Psif([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, p[0], ['pl', 'expc', 'c'])
#     p0 = np.array([0., 0.])
#     #print(cmod.loglikelihood(p0)); return
#     #fqlag.misc.check_grad(cmod, p0); return
    
#     c = maximize(cmod, p0)
#     print(c[0], c[1])
#     print()
    
#     # PPsif
#     cmod = fqlag.PPsif([tarr, tarr[1:]], [rarr, sarr[1:]], [rerr, serr[1:]], fql, ['pl', 'expc', 'c'])
#     p0 = np.concatenate([p[0], c[0]])
#     p0 = np.concatenate([p[0], [-0.8, 1]])
#     #print(cmod.loglikelihood(p0)); return
#     #fqlag.misc.check_grad(cmod, p0); return

#     c = maximize(cmod, p0)
#     print(c[0], c[1])


if __name__ == '__main__':
    
    parser  = ARG.ArgumentParser(
        description="Basic runs to check the different classes work",            
        formatter_class=ARG.ArgumentDefaultsHelpFormatter) 

    f_list = [k for k in locals() if k.startswith('test_')]
    parser.add_argument("func", metavar="func", type=str,
                        help=f'The name of the method to call. The list includes: ({", ".join(f_list)})')
    
    # process arguments #
    args = parser.parse_args()

    run_f = locals()[args.func]
    _print(f'Running {args.func}')
    run_f()
