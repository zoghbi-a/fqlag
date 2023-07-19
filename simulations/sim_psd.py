#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse as ARG
import matplotlib.pyplot as plt

import aztools as az

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))
import fqlag

def simulate_light_curves(**kwargs):
    """Simulate light curves"""

    # set defaults; we set them explictely so they are stored in kwargs
    # which is part of the output
    kwargs.setdefault('input_psd', ['powerlaw', [1e-4, -2]])
    kwargs.setdefault('seed', 3984023)
    kwargs.setdefault('n', 2**8)
    kwargs.setdefault('dt', 1.0)
    kwargs.setdefault('mu', 100.0)
    kwargs.setdefault('nsim', 200)
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
    
    plot_lc = kwargs.get('plot_lc', False)


    # do the simulations #
    sim = az.SimLC(seed)
    sim.add_model(*input_psd)

    n0 = n if gaps is None else 2*n
    if not gaps is None:
        np.random.seed(gaps[0])
    
    lc = []
    for isim in range(nsim):
        sim.simulate(n0 * nMult, dt, mu, norm='rms')
        tarr, yarr = sim.lcurve[0][:n0], sim.lcurve[1][:n0]
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
        yarr = sim.add_noise(yarr, gaussNoise, seed=noiseSeed, deltat=dt)
        if gaussNoise is None:
            yerr = ((yarr * dt)**0.5 / dt)
        else:
            yerr = yarr * 0 + gaussNoise
        lc.append([tarr, yarr, yerr])

    psd_model = np.array(sim.normalized_psd)[:,1:-1]
    extra = {'psd_model': psd_model}
    extra.update(kwargs)
    # lc.shape: (nsim, 3, n)
    ## -- plot lc only    
    if plot_lc:
        for t,r,re in lc[:5]:
            plt.errorbar(t, r, re)
        plt.savefig('figures/tmp.png')
        exit(0)
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


def fit_log_psd(fql, lc, sim_extra, suff, Dt=None, buff=False):
    """Calculated log psd for a set of simulated light curves
    
    Do all fits in log space

    Args:
        fql: array of frequency boundaries
        lc: array of simulated light curves of shape (nsim, 3, nt)
        sim_extra: dict from simulate_light_curves
        suff: e.g. '1' so files are saved as psd__1.*
        Dt: if not None, apply aliasing correction 
        buff: if True, ignore buffer at low/high; if 1; ignore low
        if 2, ignore high

    """
    az.misc.set_fancy_plot(plt)
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
    
    # remove buff freq if needed before plotting
    nfq = (fits.shape[1]-1)//2
    if isinstance(buff, bool) and buff == True:
        fits = np.hstack((fits[:,1:nfq-1], fits[:,nfq+1:-2], fits[:,[-1]]))
        fql = fql[1:-1]
    elif isinstance(buff, int) and buff == 1:
        fits = np.hstack((fits[:,1:nfq-1], fits[:,nfq+1:]))
        fql = fql[1:]
    elif isinstance(buff, int) and buff == 2:
        fits = np.hstack((fits[:,:nfq-1], fits[:,nfq+1:-2], fits[:,[-1]]))
        fql = fql[:-1]
    
    plot_psd(fits, fql, psd_model)
    os.system('mkdir -p figures')
    plt.savefig('figures/psd__%s.png'%suff)

    os.system('mkdir -p npz')
    np.savez('npz/psd__%s.npz'%suff, fits=fits, fql=fql, sim_data=sim_extra)


def psd_1(**kwargs):
    """PSD
    psd:    powerlaw
    log:    yes
    gaps:   no
    alias:  no
    extend: no
    noise:  1.0
    lfact : 1
    """
    n   = 2**8
    dt  = 1.0
    fql = np.logspace(np.log10(1./(n*dt)), np.log10(0.5*dt), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt)
    fit_log_psd(fql, lc, extra, '1', Dt=None)


def psd_2(**kwargs):
    """PSD
    psd:    powerlaw
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (included)
    noise:  1.0
    lfact : 1
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt)
    fit_log_psd(fql, lc, extra, '2', Dt=None)


def psd_3(**kwargs):
    """PSD
    psd:    powerlaw
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (ignored)
    noise:  1.0
    lfact : 1
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt)), np.log10(0.5*dt), 6)
    fql = np.concatenate([[fql[0]/nexd], fql, [fql[-1]*nexd]])
    lc, extra = simulate_light_curves(n=n, dt=dt)
    fit_log_psd(fql, lc, extra, '3', Dt=None, buff=True)


def psd_4(**kwargs):
    """PSD
    psd:    powerlaw
    log:    yes
    gaps:   no
    alias:  no
    extend: 4 (included)
    noise:  1.0
    lfact : 1
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt)
    fit_log_psd(fql, lc, extra, '4', Dt=None)


def psd_5(**kwargs):
    """PSD
    psd:    flatter powerlaw [1e-4, -1]
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (included)
    noise:  1.0
    lfact : 1
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt, input_psd=['powerlaw', [1e-4, -1]])
    fit_log_psd(fql, lc, extra, '5', Dt=None)


def psd_6(**kwargs):
    """PSD
    psd:    steeper powerlaw [1e-4, -2.5]
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (included)
    noise:  1.0
    lfact : 1
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt, input_psd=['powerlaw', [1e-4, -2.5]])
    fit_log_psd(fql, lc, extra, '6', Dt=None)


def psd_7(**kwargs):
    """PSD
    psd:    powerlaw
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (included)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt, nMult=4)
    fit_log_psd(fql, lc, extra, '7', Dt=None)


def psd_8(**kwargs):
    """PSD
    psd:    steeper powerlaw [1e-6, -2.5]
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (included)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt, nMult=4, input_psd=['powerlaw', [1e-6, -2.5]])
    fit_log_psd(fql, lc, extra, '8', Dt=None)


def psd_9(**kwargs):
    """PSD
    psd:    steeper powerlaw [1e-6, -2.5]
    log:    yes
    gaps:   no
    alias:  no
    extend: 5 (included)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 5
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(n=n, dt=dt, nMult=4, input_psd=['powerlaw', [1e-6, -2.5]])
    fit_log_psd(fql, lc, extra, '9', Dt=None)


def psd_10(**kwargs):
    """PSD
    psd:    steeper powerlaw [1e-6, -2.5]
    log:    yes
    gaps:   no
    alias:  no
    extend: 5 (extra)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 5
    fql = np.logspace(np.log10(1/(n*dt)), np.log10(0.5*dt), 5)
    fql = np.concatenate([[fql[0]/nexd], fql, [fql[-1]*nexd]])
    lc, extra = simulate_light_curves(n=n, dt=dt, nMult=4, input_psd=['powerlaw', [1e-6, -2.5]])
    fit_log_psd(fql, lc, extra, '10', Dt=None, buff=True)

    
def psd_11(**kwargs):
    """PSD
    psd:    broken powerlaw [1e-6, -1, -3, 3e-3]
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (include)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(
        n=n, dt=dt, nMult=4,
        input_psd=['broken_powerlaw', [1e-7, -1, -3, 3e-3]]
    )
    fit_log_psd(fql, lc, extra, '11', Dt=None)


def psd_12(**kwargs):
    """PSD
    psd:    broken powerlaw [1e-6, -1, -2, 3e-3]
    log:    yes
    gaps:   no
    alias:  no
    extend: 2 (include)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(
        n=n, dt=dt, nMult=4,
        input_psd=['broken_powerlaw', [7e-5, -1, -2, 3e-3]]
    )
    fit_log_psd(fql, lc, extra, '12', Dt=None)

    
def psd_13(**kwargs):
    """PSD
    psd:    broken powerlaw [1e-6, -1, -2, 3e-3]
    log:    yes
    gaps:   yes, same
    alias:  no
    extend: 2 (include)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(
        n=n, dt=dt, nMult=4,
        input_psd=['broken_powerlaw', [7e-5, -1, -2, 3e-3]],
        gaps=[3372, 50], sameGap=True
    )
    fit_log_psd(fql, lc, extra, '13', Dt=None)
    
def psd_14(**kwargs):
    """PSD
    psd:    broken powerlaw [1e-6, -1, -2, 3e-3]
    log:    yes
    gaps:   yes, different
    alias:  no
    extend: 2 (include)
    noise:  1.0
    lfact : 4
    """
    n   = 2**8
    dt  = 1.0
    nexd = 2
    fql = np.logspace(np.log10(1/(n*dt*nexd)), np.log10(0.5*dt*nexd), 6)
    lc, extra = simulate_light_curves(
        n=n, dt=dt, nMult=4,
        input_psd=['broken_powerlaw', [7e-5, -1, -2, 3e-3]],
        gaps=[3372, 50], sameGap=False
    )
    fit_log_psd(fql, lc, extra, '14', Dt=None)
    


if __name__ == '__main__':
    
    parser  = ARG.ArgumentParser(
        description="Run simulations for the PSD calculation",            
        formatter_class=ARG.ArgumentDefaultsHelpFormatter) 

    parser.add_argument('-p', '--psd', metavar='psd', type=int,
            help='Run psd simulation number psd')


    # process arguments #
    args = parser.parse_args()

    if not os.path.exists(f'npz/psd__{args.psd}.npz') or not os.path.exists(f'figures/psd__{args.psd}.png'):
        psd_f = locals()[f'psd_{args.psd}']
        psd_f()
    else:
        print('files exist; nothing to do')
