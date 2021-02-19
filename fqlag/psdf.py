import numpy as np
from .base import FqLagBin

## --- Functions that can fit to the PSD --- ##
## This are the function that can be directly 
## used to model the PSD/Covariance. The user
## may define their own model and passing it
## to Psdf below

# ---- powerlaw ---- #
# pars: norm, indx
def pfunc__pl(f, p):
    a, b = np.exp(p[0]), p[1]
    return a * f**b   

def pderv__pl(f, p):
    return pfunc__pl(f, p) * np.array([f*0+1., np.log(f)])

    
# ---- const ---- #
# pars: const
pfunc__c = lambda f,p: f*0+p[0]
pderv__c = lambda f,p: np.array([f*0+1])


# ---- log-constant ---- #
# pars: log-psd
pfunc__logc = lambda f,p: f*0+np.exp(p[0])
pderv__logc = lambda f,p: np.array([pfunc__expc(f,p)])


# ---- bending powerlaw ---- #
# pars: norm, index, break
def pfunc__bpl(f, p):
    a, b, c = np.exp(p[0]), p[1], np.exp(p[2])
    return (a/f) * 1./(1 + (f/c)**(-b-1))

def pderv__bpl(f, p):
    a, b, c = np.exp(p[0]), p[1], np.exp(p[2])
    foc = f/c
    return np.array([
        pfunc__bpl(f, p), 
        a*foc**(-1-b) * np.log(foc) / (f*(1+foc**(-1-b))**2), 
        ((-1-b)*a/c * (foc)**(-2-b)) / (1+foc**(-1-b))**2
    ])


# ---- Lorentzian ---- #
# pars: norm, fq_cent, fq_sigma
def pfunc__lor(f, p):
    a, b, c = np.exp(p[:3])
    return a * (c/(2*np.pi)) / ( (f-b)**2 + (c/2)**2 )

def pderv__lor(f, p):
    a, b, c = np.exp(p[:3])
    return np.array([
        pfunc__lor(f, p), 
        a*b*c*(f-b)/(np.pi*((-b+f)**2+(c*c/4))**2),
        (-a*c**3/(4*np.pi*((-b+f)**2+(c*c/4))**2) +
            a*c/(2*np.pi*((-b+f)**2+(c*c/4))) )
    ])


# ---- zero-centered Lorentzian ---- #
# pars: norm, fq_sigma
def pfunc__lor0(f, p):
    a, c = np.exp(p[:2])
    return a * (c/(2*np.pi)) / ( f**2 + (c/2)**2 )

def pderv__lor0(f, p):
    a, c = np.exp(p[:3])
    return np.array([
        pfunc__lor0(f, p), 
        (-a*c**3/(4*np.pi*((f**2)+(c*c/4))**2) +
        a*c/(2*np.pi*((f**2)+(c*c/4))) )
    ])

def identify_model(model):
    """Identify which built-in model is to be used.

    The models are:
    - pl: powerlaw with parameters: (log)norm, index
    - c: constant with parameter: const
    - logc: a constant in log-units with parameters: const(log) 
    - 'bpl': bending powerlaw with parameters: 
            (log)norm, index, (log)bend_fre
    - 'lor': lorentzian with parameters: 
            (log)norm, (log)fq_cent, (log)fq_sigma
    - 'lor0': zero-centered lorentzian with parameters:
            (log)norm, (log)fq_sigma
    
    Args:
        model: A string for the name of the model
            or a sum of models. e.g: pl or pl+bpl

    Returns:
        (pfunc, pderv, npar) where `pfunc` is a python method 
        to calculate psd, `pderv` is a python method to calculate
        the derivative, and `npar` is the number of parameters each 
        of the method take

    """
    
    # a create a list of built-in model names
    # with their defining function/derivative 
    models = {}
    models['pl'] = [pfunc__pl, pderv__pl, 2] 
    models['c'] = [pfunc__c, pderv__c, 1]
    models['logc'] = [pfunc__logc, pderv__logc, 1]
    models['bpl'] = [pfunc__bpl, pderv__bpl, 3]
    models['lor'] = [pfunc__lor, pderv__lor, 3]
    models['lor0'] = [pfunc__lor0, pderv__lor0, 2]
    

    ## -- work out the model -- ##
    if '+' in model:
        # The model is a sum of a few built-in models
        # TODO: This doesn't work all the time.
        # It needs to be checked properly
        mods = [i.strip() for i in model.split('+')]
        mods = [models[i] for i in mods]
        npar = np.sum([m[2] for m in mods])
        ipars,ip = [], 0
        for m in mods:
            ipars.append(list(range(ip, ip+m[2])))
            ip += m[2]
        
        def pfunc(f, p):
            return np.sum([m[0](f, p[ipars[im]]) for im,m in enumerate(mods)], axis=0)
        def pderv(f, p):
            return np.concatenate([m[1](f, p[ipars[im]]) for im,m in enumerate(mods)], axis=0)
        
    else:
        # we have a single built-in model
        if not model in models.keys():
            raise ValueError('model %s is not part of built-in models: [%s]'%(
                model, ', '.join(models.keys())))
        pfunc, pderv, npar = models[model]
        
    return pfunc, pderv, npar



class Psdf(FqLagBin):
    """Model for calculating Power Spectral Density (PSD) 
        with a functional form. e.g. powerlaw, lorentzian.

    The calculation is dome by splitting the frequency domain in NFQ
    bins, where the psd values in the bins are calulated using the 
    selected function.

    This class defines the covariance kernel, and the calculations are done
    in FqLagBin that Psd inherits from.

    Args:
        tarr: a numpy array giving the time axis of the light curve.
        yarr: a numpy array giving the count rate or flux
        yerr: a numpy array giving the 1-sigma measurement uncertainity
            in the count rate or flux.
        fql: [fq_min, fq_max] the minimum and maximum of the frequency
            range to be modeled. This should be wider than what the light cuve
            contains
        model: string of a built-in model. See @identify_model function for 
            a list of built-in functions.
        dt: sampling time of the light curves. If given, corrections to sampling
            bias is applied, otherwise, we don't apply it.
        NFQ: The number of bins in the frequency grid used in the calculations.
            This is the 'resolution' of the model.
    """
    
    def __init__(self, tarr, yarr, yerr, fql, model='pl', dt=None, NFQ=8):
        # Define the frequency grid and initialize parent class
        self.NFQ = NFQ
        fqL      = np.logspace(np.log10(fql[0]), np.log10(fql[1]), NFQ)
        self.fq  = (fqL[1:] + fqL[:-1]) / 2.
        self.fqL = np.array(fqL)
        
        super(Psdf, self).__init__(tarr, yarr, yerr, fqL, dt)
        self.norm = self.mu**2
        
        
        pfunc, pderv, npar = identify_model(model)
        self.psd_func = pfunc
        self.psd_derv = pderv


    def covariance(self, pars):
        """Covariance kernel function for modeling the power spectrum.
        
        The model parameters are the function parameters for psd
        (see @identify_model for a list of built-in models).
        The psd is in rms normalization. 

        cov = sum_i(PsdFunc(f_i) * I_s[:,:,i]), with i=0..(NFQ-1)
        and I_s is the Fourier integral defined in FqLagBin
        
        
        Args:
            pars: parameters of the function that defines the psd/covariance.

        Returns:
            A covariance matrix of shape: (self.n, self.n)

        """
        # calculate the psd and normlize it
        psd  = self.psd_func(self.fq, pars)
        psd *= self.norm

        return np.sum(psd * self.I_s, -1)

    def covariance_derivative(self, pars):
        """First Derivative of the covariance kernel function 
        for modeling the power spectrum.

        This calculates the derivative of covariance values with respect
        to each model parameter
        

        Args:
            pars: parameters of the covariance kernel.

        Returns:
            a matrix of first derivatives with shape (npar, self.n, self.n)"""
        psdD  = self.psd_derv(self.fq, pars)
        psdD *= self.norm
        return np.sum(self.I_s * psdD[:,None,None,:], -1)


    