#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse as ARG
import pylab as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.')))
    
import fqlag
from sim_psd import simulate_light_curves

try:
    import aztools as az
except:
    raise ImportError(('aztools was not found. Please download from '
        'https://github.com/zoghbi-a/aztools'))





def plot_psdf(pfits, fits, psd_model, input_pars=None):
    """Summarize psd simulation"""
    # pfits.shape: (nsim, 2*npar+1)
    # fits.shape: (nsim+1, nfq)


    mod_fq, mod_psd = psd_model

    pars, perr  = np.split(pfits[:,:-1], 2, axis=1)
    npar = pars.shape[1]


    fit_q  = np.quantile(pars, [0.5, 0.1587, 0.8413], 0)
    fit_p, fit_sd = np.median(pars, 0), np.std(pars, 0)
    fit_e  = np.mean(perr, 0)

    m_fq, m_mod  = fits[0], fits[1:]
    m_mod = np.log(m_mod)
    m_q  = np.quantile(m_mod, [0.5, 0.1587, 0.8413], 0)
    m_p, m_sd = np.median(m_mod, 0), np.std(m_mod, 0)


    fig, ax = plt.subplots(1,1+pars.shape[1],figsize=(10,4))

    ax[0].semilogx(mod_fq, mod_psd, label='model', color='C0')
    ax[0].fill_between(m_fq, m_q[1], m_q[2], alpha=0.5, color='C1')


    for ii in range(npar):
        h = ax[ii+1].hist(pars[:,ii], 20, density=1, alpha=0.5)
        ax[ii+1].plot([fit_q[0,ii]]*2, [0, h[0].max()])
        if not input_pars[ii] is None:
            ax[ii+1].plot([input_pars[ii]]*2, [0, h[0].max()], lw=3)
    plt.tight_layout()


def fit_psdf(fql, model, lc, sim_extra, suff, Dt=None, input_pars=None):
    """Calculated log psd for a set of simulated light curves

    Args:
        fql: array of frequency boundaries
        model: [mod, p0] mod: string for built-in model in psdf, 
            p0: starting parameters for the model
        lc: array of simulated light curves of shape (nsim, 3, nt)
        sim_extra: dict from simulate_light_curves
        suff: e.g. '1' so files are saved as psdf__1.*
        Dt: if not None, apply aliasing correction 
        input_pars: a list of input parameters used in the simulaion
            to compare the results to, Use None for parameter to skip.
            Used mainly for plotting

    """

    mod, p0 = model
    p0 = np.array(p0)
    fits, pfits = [], []
    for tarr,yarr,yerr in lc:
        pmod = fqlag.Psdf(tarr, yarr, yerr, fql, mod, dt=Dt)
        pfit = fqlag.misc.maximize(pmod, p0)
        pfits.append(np.concatenate([pfit[0], pfit[1], [-pfit[2].fun]]))
        fits.append(pmod.psd_func(pmod.fq, pfit[0]))
        
    # shape: nsim, 2*npar+1 (pars, perr, loglikelihod)
    pfits = np.array(pfits)
    fits  = np.array(fits)
    fits  = np.r_[[pmod.fq], fits]


    psd_model = sim_extra['psd_model']
    psd_model[1] = np.log(psd_model[1])
    plot_psdf(pfits, fits, psd_model, input_pars)
    os.system('mkdir -p figures')
    plt.savefig('figures/psdf__%s.png'%suff)

    os.system('mkdir -p npz')
    np.savez('npz/psdf__%s.npz'%suff, fits=fits, fql=fql, sim_data=sim_extra)


def psdf_1(**kwargs):
    """input powerlaw psd; no gaps, no alias, noleak, gauss noise, fit with pl,
    no extended frequency, NFQ=8"""

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.array([1./(dt*n), 0.5/dt])

    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100)

    model = ['pl', [-5, -2]]
    inP  = extra['input_psd'][1]
    inP[0] = np.log(inP[0])
    fit_psdf(fql, model, lc, extra, '1', input_pars=inP)

def psdf_2(**kwargs):
    """input powerlaw psd; no gaps, no alias, noleak, gauss noise, fit with pl,
    EXTEND frequency, NFQ=8"""

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.array([.5/(dt*n), 1./dt])

    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100)

    model = ['pl', [-5, -2]]
    inP  = extra['input_psd'][1]
    inP[0] = np.log(inP[0])
    fit_psdf(fql, model, lc, extra, '2', input_pars=inP)


def psdf_3(**kwargs):
    """input powerlaw psd; no gaps, no alias, noleak, gauss noise, fit with pl,
    EXTEND frequency, NFQ=24"""

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.array([.5/(dt*n), 1./dt])

    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100)

    model = ['pl', [-5, -2]]
    inP  = extra['input_psd'][1]
    inP[0] = np.log(inP[0])
    fit_psdf(fql, model, lc, extra, '3', input_pars=inP)


def psdf_4(**kwargs):
    """input bkn pl psd; no gaps, no alias, noleak, gauss noise, fit with bpl,
    EXTEND frequency"""

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.array([.5/(dt*n), 1./dt])

    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100,
            input_psd=['broken_powerlaw', [1e-4, -1, -2, 3e-2]])

    model = ['bpl', [-5, -2, -3]]
    inP  = extra['input_psd'][1]
    inP = [np.log(inP[0]), inP[2], np.log(inP[3])]
    fit_psdf(fql, model, lc, extra, '4', input_pars=inP)


def psdf_5(**kwargs):
    """input bpl psd; no gaps, no alias, noleak, gauss noise, fit with bpl,
    EXTEND frequency"""

    # fqlag parameters #
    n       = 2**8
    dt      = 1.0
    fql     = np.array([.5/(dt*n), 1./dt])

    lc, extra = simulate_light_curves(n=n, dt=dt, nsim=100,
            input_psd=['bending_powerlaw', [1e-4, -2, 3e-3]])

    model = ['bpl', [-5, -2, -5]]
    inP  = extra['input_psd'][1]
    inP = [np.log(inP[0]), inP[1], np.log(inP[2])]
    fit_psdf(fql, model, lc, extra, '5', input_pars=inP)


if __name__ == '__main__':
    
    parser  = ARG.ArgumentParser(
        description="Run simulations for the PSD calculation",            
        formatter_class=ARG.ArgumentDefaultsHelpFormatter) 


    parser.add_argument('--psdf_1', action='store_true', default=False,
            help="psd modeling with pl model simulation.")
    parser.add_argument('--psdf_2', action='store_true', default=False,
            help="psd modeling with pl, extend freq.")
    parser.add_argument('--psdf_3', action='store_true', default=False,
            help="psd modeling with pl, extend freq. nfq=24")
    parser.add_argument('--psdf_4', action='store_true', default=False,
            help="psd modeling with pl, extend freq.. bkn pl input")
    parser.add_argument('--psdf_5', action='store_true', default=False,
            help="psd modeling with pl, extend freq.. bpl input")


    # process arguments #
    args = parser.parse_args()


    # powerlaw psd, log; no gaps, no alias, noleak, gauss noise, no extra freq #
    if args.psdf_1: psdf_1()

    # powerlaw psd, log; no gaps, no alias, noleak, gauss noise, EXTENDED freq #
    if args.psdf_2: psdf_2()

    # powerlaw psd, log; no gaps, no alias, noleak, gauss noise, EXTENDED freq; nfq=24 #
    if args.psdf_3: psdf_3()

    # bkn pl psd, log; no gaps, no alias, noleak, gauss noise, EXTENDED freq; #
    if args.psdf_4: psdf_4()

    # bpl psd, log; no gaps, no alias, noleak, gauss noise, EXTENDED freq; #
    if args.psdf_5: psdf_5()
