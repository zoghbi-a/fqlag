import numpy as np

class multiFqLagBin:
    """A class for fitting multiple light curves simultaneously

    The input a list of models that need to be satisfied simultaneously.
    The total loglikelihood is the sum of the loglikelihood functions from
    the input models

    Args:
        models: a list of models. Typicall a list of psd models or a list of
            psi or cxd models.

    """
    
    def __init__(self, models):
        self.nmod = len(models)
        self.mods = [m for m in models]
        
    def loglikelihood(self, pars):
        """Calculate the loglikelihood from the combination of the input models
        See individual models ofr details of call.

        loglikelihood = SUM([m.loglikelihood(pars) for m in self.mods])

        Args:
            pars: parameters of the model

        Returns:
            a single number giving the log-likelihood
        

        """
        return np.sum([m.loglikelihood(pars) for m in self.mods])


    def loglikelihood_derivative(self, pars, calc_fisher=True):
        """Calculate the derivative of the loglikelihood function by summing
        the derivatives of the constituent models.


        Args:
            pars: parameters of the model as a numpy array
            calc_fisher: Calculate the fisher matrix to approximate the
                the Hessian. This may or not be needed depending on the
                optimizition algorithm used. Default is True.

        Returns:
            if calc_fisher is True:
                return logLikelihood, gradient_array
            else:
                return: logLikelihood, gradient_array, fisher_matrix

        """
        lgh = [m.loglikelihood_derivative(pars, calc_fisher) for m in self.mods]

        logLike = np.sum([l[0] for l in lgh])
        grad    = np.sum([l[1] for l in lgh], axis=0)

        if not calc_fisher:
            return logLike, grad

        fisher  = np.sum([l[2] for l in lgh], axis=0)
        return logLike, grad, fisher
