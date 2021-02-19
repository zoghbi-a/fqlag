#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse as ARG
import pylab as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import fqlag

try:
    import aztools as az
except:
    raise ImportError(('aztools was not found. Please download from '
        'https://github.com/zoghbi-a/aztools'))


def simulate_light_curves(**kwargs):
    """Simulate light curves"""

    # set defaults; we set them explictely so they are stored in kwargs
    # which is part of the output
    kwargs.setdefault('input_psd', ['powerlaw', [1e-4, -2]])
    kwargs.setdefault('seed', 3984023)
    kwargs.setdefault('n', 2**8)
    kwargs.setdefault('dt', 1.0)
    kwargs.setdefault('mu', 100.0)
    kwargs.setdefault('nsim', 100)
    kwargs.setdefault('gaussNoise', 1.0)
    kwargs.setdefault('nMult', 1)
    kwargs.setdefault('noiseSeed', 237)
    kwargs.setdefault('gaps', None)
    kwargs.setdefault('sameGap', False)


    input_psd   = kwargs['input_psd']
    seed        = kwargs['seed']
    n           = kwargs['n']
    dt          = kwargs['dt']
    mu          = kwargs['mu']
    nsim        = kwargs['nsim']
    gaussNoise  = kwargs['gaussNoise']
    nMult       = kwargs['nMult']
    noiseSeed   = kwargs['noiseSeed']
    # if not None; gaps = [seed, period (sec)]
    gaps        = kwargs['gaps']
    sameGap     = kwargs['sameGap']
    


    # do the simulations #
    sim = az.SimLC(seed)
    sim.add_model(*input_psd)

    n0 = n if gaps is None else 2*n
    if not gaps is None:
        np.random.seed(gaps[0])
    
    lc = []
    for isim in range(nsim):
        sim.simulate(n0 * nMult, dt, mu, 'rms')
        tarr, yarr = sim.t[:n0], sim.x[:n0]
        if not gaps is None:
            prob = np.cos(2*np.pi*tarr / gaps[1])**2
            prob = prob / prob.sum()
            if isim == 0:
                idx = np.sort(np.random.choice(np.arange(n0), n, False, prob))
            elif not sameGap:
                idx = np.sort(np.random.choice(np.arange(n0), n, False, prob))
            else: 
                pass
            tarr, yarr = tarr[idx], yarr[idx]
        yarr = sim.add_noise(yarr, gaussNoise, seed=noiseSeed, dt=dt)
        if gaussNoise is None:
            yerr = ((yarr * dt)**0.5 / dt)
        else:
            yerr = yarr * 0 + gaussNoise
        lc.append([tarr, yarr, yerr])

    psd_model = np.array(sim.normalized_psd)[:,1:-1]
    extra = {'psd_model': psd_model}
    extra.update(kwargs)
    # lc.shape: (nsim, 3, n)
    return np.array(lc), extra


def plot_psd(fits, fql, psd_model):
    """Summarize psd simulation"""

    mod_fq, mod_psd = psd_model

    pars, perr  = np.split(fits[:,:-1], 2, axis=1)
    fit_fq = np.exp((np.log(fql[1:])+np.log(fql[:-1]))/2.0)

    fit_q  = np.quantile(pars, [0.5, 0.1587, 0.8413], 0)
    fit_p, fit_sd = np.median(pars, 0), np.std(pars, 0)
    fit_e  = np.mean(perr, 0)

    fig, ax = plt.subplots(1,2,figsize=(10,4))

    ax[0].semilogx(mod_fq, mod_psd, label='model', color='C0')
    ax[0].plot(fit_fq, fit_q[0], '-', color='C1', label='quantile')
    ax[0].fill_between(fit_fq, fit_q[1], fit_q[2], alpha=0.5, color='C1')
    ax[0].plot(fit_fq, fit_q[0]-fit_e, '-.', color='C2', lw=0.5, label='meanErr')
    ax[0].plot(fit_fq, fit_q[0]+fit_e, '-.', color='C2', lw=0.5)
    ax[0].errorbar(fit_fq, fit_p, fit_sd, fmt='o', color='C3', label='mean/std')
    ax[0].legend()
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('log power')
    ax[0].set_ylim([mod_psd.min()-1, mod_psd.max()+1])


    for ii in range(pars.shape[1]):
        h = ax[1].hist(fits[:,ii], 20, density=1, alpha=0.5)
        fm = mod_psd[np.argmin(np.abs(mod_fq - fit_fq[ii]))]
        ax[1].plot([fm, fm], [0, h[0].max()])
    plt.tight_layout()


def fit_log_psd(fql, lc, sim_extra, suff, Dt=None):
    """Calculated log psd for a set of simulated light curves

    Args:
        fql: array of frequency boundaries
        lc: array of simulated light curves of shape (nsim, 3, nt)
        sim_extra: dict from simulate_light_curves
        suff: e.g. '1' so files are saved as psd__1.*
        Dt: if not None, apply aliasing correction 

    """

    p0 = np.zeros_like(fql[1:]) - 1
    fits = []
    for tarr,yarr,yerr in lc:
        pmod = fqlag.Psd(tarr, yarr, yerr, fql, dt=Dt, log=True)
        pfit = fqlag.misc.maximize(pmod, p0)
        if np.any(pfit[0] < -10) or np.any(pfit[1] > 10):
            p0 = pfit[0]
            p0[p0<-10] = -2
            p0[pfit[1]>10] = -2
            pfit = fqlag.misc.maximize(pmod, p0)
        fits.append(np.concatenate([pfit[0], pfit[1], [-pfit[2].fun]]))
    # shape: nsim, 2*nfq+1 (pars, perr, loglikelihod)
    fits = np.array(fits)


    psd_model = sim_extra['psd_model']
    psd_model[1] = np.log(psd_model[1])
    plot_psd(fits, fql, psd_model)
    os.system('mkdir -p figures')
    plt.savefig('figures/psd__%s.png'%suff)

    os.system('mkdir -p npz')
    np.savez('npz/psd__%s.npz'%suff, fits=fits, fql=fql, sim_data=sim_extra)


def psd_1(**kwargs):
    """powerlaw psd, log; no gaps, no alias, noleak, gauss noise, no extra freq"""

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(1./(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100)

    fit_log_psd(fql, lc, extra, '1')


def psd_2(**kwargs):
    """powerlaw psd, log; no gaps, no alias, noleak, gauss noise
    Frequencies extended
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.5/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100)

    fit_log_psd(fql, lc, extra, '2')


def psd_3(**kwargs):
    """powerlaw psd, log; no gaps, no alias, noleak, gauss noise
    Frequencies extended; smaller extension
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.8/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100)

    fit_log_psd(fql, lc, extra, '3')


def psd_4(**kwargs):
    """powerlaw psd, log; no gaps, no alias, noleak, gauss noise
    small fq extension; steeper psd
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.8/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, 
                            input_psd=['powerlaw', [1e-5, -3]])

    fit_log_psd(fql, lc, extra, '4')


def psd_5(**kwargs):
    """powerlaw psd, log; no gaps, no alias, noleak, gauss noise
    small fq extension; flatter psd
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.8/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, 
                            input_psd=['powerlaw', [1e-4, -1]])

    fit_log_psd(fql, lc, extra, '5')


def psd_6(**kwargs):
    """powerlaw psd, log; no gaps, no alias, gauss noise
    small fq extension; steeper psd; x4 read leak
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.8/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=4)

    fit_log_psd(fql, lc, extra, '6')


def psd_7(**kwargs):
    """powerlaw psd, log; no gaps, no alias, gauss noise
    0.5 fq extension; x4 red leak
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.5/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=4)

    fit_log_psd(fql, lc, extra, '7')


def psd_8(**kwargs):
    """powerlaw psd, log; no gaps, no alias, gauss noise
    0.5 fq extension; x8 red leak
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.5/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=8)

    fit_log_psd(fql, lc, extra, '8')


def psd_9(**kwargs):
    """powerlaw psd, log; no gaps, no alias, gauss noise
    0.5 fq extension; x8 red leak; rescale lc mean
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.5/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=8)
    lc[:,1] = lc[:,1] - lc[:,1].mean(1)[:,None] + lc[:,1].mean()

    fit_log_psd(fql, lc, extra, '9')


def psd_10(**kwargs):
    """powerlaw psd, log; no gaps, no alias, gauss noise
    0.5 fq extension; x4 red leak; rescale lc mean
    different psd (changing index only produces unrealistic lc -> redcue norm too)
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.5/(dt*n)), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=4,
                    input_psd=['powerlaw', [1e-6, -3]])
    lc[:,1] = lc[:,1] - lc[:,1].mean(1)[:,None] + lc[:,1].mean()

    fit_log_psd(fql, lc, extra, '10')


def psd_11(**kwargs):
    """bkn powerlaw psd, log; no gaps, no alias, gauss noise
    0.5 fq extension; x4 red leak; rescale lc mean; broken pl psd
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.logspace(np.log10(0.5/n), np.log10(0.5*dt), 6)

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=4,
            input_psd=['broken_powerlaw', [1e-6, -1, -3, 3e-3]])
    lc[:,1] = lc[:,1] - lc[:,1].mean(1)[:,None] + lc[:,1].mean()

    fit_log_psd(fql, lc, extra, '11')



def psd_12(**kwargs):
    """bkn powerlaw psd, log; Gaps, no alias, Lager gauss noise
    0.5 fq extension; x4 red leak; rescale lc mean; broken pl psd
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=4,
            input_psd=['broken_powerlaw', [1e-4, -1, -2, 1e-3]], gaussNoise=1,
            gaps=[3372, 50])
    
    fql     = np.logspace(np.log10(0.5/(lc[0,0,-1]-lc[0,0,0])), np.log10(0.5*dt), 6)

    lc[:,1] = lc[:,1] - lc[:,1].mean(1)[:,None] + lc[:,1].mean()

    fit_log_psd(fql, lc, extra, '12')


def psd_13(**kwargs):
    """bkn powerlaw psd, log; Same Gaps, no alias, Lager gauss noise
    0.5 fq extension; x4 red leak; rescale lc mean; broken pl psd
    """

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0

    
    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100, nMult=4,
            input_psd=['broken_powerlaw', [1e-4, -1, -2, 1e-3]], gaussNoise=1,
            gaps=[3372, 50], sameGap=True)
    
    fql     = np.logspace(np.log10(0.5/(lc[0,0,-1]-lc[0,0,0])), np.log10(0.5*dt), 6)

    lc[:,1] = lc[:,1] - lc[:,1].mean(1)[:,None] + lc[:,1].mean()

    fit_log_psd(fql, lc, extra, '13')


if __name__ == '__main__':
    
    parser  = ARG.ArgumentParser(
        description="Run simulations for the PSD calculation",            
        formatter_class=ARG.ArgumentDefaultsHelpFormatter) 


    parser.add_argument('--psd_1', action='store_true', default=False,
            help="Simple psd simulation.")
    parser.add_argument('--psd_2', action='store_true', default=False,
            help="Simple psd simulation. Extended freq")
    parser.add_argument('--psd_3', action='store_true', default=False,
            help="Simple psd simulation. Extended freq")
    parser.add_argument('--psd_4', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; different psd")
    parser.add_argument('--psd_5', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; different psd")

    parser.add_argument('--psd_6', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; red leak")
    parser.add_argument('--psd_7', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; red leak")
    parser.add_argument('--psd_8', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; red leak")
    parser.add_argument('--psd_9', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; red leak; re-mean")
    parser.add_argument('--psd_10', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; red leak; re-mean, diff psd")
    parser.add_argument('--psd_11', action='store_true', default=False,
            help="Simple psd simulation. Extended freq; red leak; re-mean, bpl psd")
    parser.add_argument('--psd_12', action='store_true', default=False,
            help="Extended freq; red leak; re-mean, bpl psd; Gaps")
    parser.add_argument('--psd_13', action='store_true', default=False,
            help="Extended freq; red leak; re-mean, bpl psd; same Gap")


    # process arguments #
    args = parser.parse_args()


    # powerlaw psd, log; no gaps, no alias, noleak, gauss noise, no extra freq #
    if args.psd_1: psd_1()

    # powerlaw psd, log; no gaps, no alias, noleak, gauss noise, extended freq #
    if args.psd_2: psd_2()

    # powerlaw psd, log; no gaps, no alias, noleak, gauss noise, extended freq #
    if args.psd_3: psd_3()

    # diff powerlaw psd, log; no gaps, no alias, noleak, gauss noise, extended freq #
    if args.psd_4: psd_4()

    # diff powerlaw psd, log; no gaps, no alias, noleak, gauss noise, extended freq #
    if args.psd_5: psd_5()

    # diff powerlaw psd, log; no gaps, no alias, leak, gauss noise, extended freq #
    if args.psd_6: psd_6()

    # powerlaw psd, log; no gaps, no alias, leak, gauss noise, 0.5 extended freq #
    if args.psd_7: psd_7()

    # powerlaw psd, log; no gaps, no alias, longer leak, gauss noise, 0.5 extended freq #
    if args.psd_8: psd_8()

    # powerlaw psd, log; no gaps, no alias, longer leak, gauss noise, 
    # 0.5 extended freq; reset the mean
    if args.psd_9: psd_9()

    # diff powerlaw psd, log; no gaps, no alias, longer leak, gauss noise, 
    # 0.5 extended freq; reset the mean
    if args.psd_10: psd_10()

    # broken powerlaw psd, log; no gaps, no alias, longer leak, gauss noise, 
    # 0.5 extended freq; reset the mean
    if args.psd_11: psd_11()

    # broken powerlaw psd, log; Gaps, no alias, longer leak, Larger gauss noise, 
    # 0.5 extended freq; reset the mean
    if args.psd_12: psd_12()

    # broken powerlaw psd, log; Same Gaps, no alias, longer leak, Larger gauss noise, 
    # 0.5 extended freq; reset the mean
    if args.psd_13: psd_13()


