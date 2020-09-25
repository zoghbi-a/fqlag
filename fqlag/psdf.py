import numpy as np
from .base import FqLagBin

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
pfunc__expc = lambda f,p: f*0+np.exp(p[0])
pderv__expc = lambda f,p: np.array([pfunc__expc(f,p)])


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
    
    models = {}
    models['pl'] = [pfunc__pl, pderv__pl, 2] 
    models['c'] = [pfunc__c, pderv__c, 1]
    models['expc'] = [pfunc__expc, pderv__expc, 1]
    models['bpl'] = [pfunc__bpl, pderv__bpl, 3]
    models['lor'] = [pfunc__lor, pderv__lor, 3]
    models['lor0'] = [pfunc__lor0, pderv__lor0, 2]
    
    ## -- work out the model -- ##
    if '+' in model:
        #raise ValueError('Not working yet')
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
        if not model in models.keys():
            raise ValueError('model %s is not supported'%model)
        pfunc, pderv, npar = models[model]
        
    return pfunc, pderv, npar



class Psdf(FqLagBin):
    
    def __init__(self, tarr, yarr, yerr, fql, model='pl', dt=None, NFQ=8):
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
        psd  = self.psd_func(self.fq, pars)
        psd *= self.norm

        return np.sum(psd * self.I_s, -1)

    def covariance_derivative(self, pars):
        psdD  = self.psd_derv(self.fq, pars)
        psdD *= self.norm
        return np.sum(self.I_s * psdD[:,None,None,:], -1)
    