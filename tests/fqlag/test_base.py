# pylint: skip-file

import os
import sys
import unittest

import numpy as np
import scipy.stats as st
from scipy.misc import derivative

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src')))
import fqlag
from fqlag import base


class TestFqLagBase(unittest.TestCase):
    """"""


    def test__init(self):
        """Input arrays have different lengths"""
        arr = np.array([0.,1,2])
        with self.assertRaises(ValueError):
            mod = base.FqLagBase(arr, arr, arr[1:])

    def test__loglikelihood(self):
        """Check for a simple covariance"""
        t = np.array([0.,1,2,3])
        y = np.array([1.,4,2,7])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        mod = base.FqLagBase(t, y, ye)
        n = len(t)
        x = y - y.mean()
        logp = -0.5*(n*np.log(2*np.pi) + np.log(0.01**n) + (x*x*100).sum())
        self.assertAlmostEqual(mod.loglikelihood([0., 0.]), logp)


    def test__loglikelihood_priors(self):
        """Check for a simple covariance"""
        t = np.array([0.,1,2,3])
        y = np.array([1.,4,2,7])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        mod = base.FqLagBase(t, y, ye)
        loglike = mod.loglikelihood([0., 0.])

        pr = st.norm(1.0, 0.5)
        mod.add_prior(0, pr)
        self.assertAlmostEqual(mod.loglikelihood([0., 0.]), loglike + pr.logpdf(0.))


    def test__loglikelihood_derivative(self):
        """Check for a simple covariance"""
        t = np.array([0.,1,2,3])
        y = np.array([1.,4,2,7])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        mod = base.FqLagBase(t, y, ye)
        p0 = np.array([0., .0])
        logp, grad = mod.loglikelihood_derivative(p0, calc_fisher=False)

        dx = 1e-7
        g0 = []
        for i in range(len(p0)):
            def f(x, pp0, ii):
                pp0[ii] = x
                return mod.loglikelihood(pp0)
            # when derivative is deprecated; copy it here from
            # https://github.com/scipy/scipy/blob/v1.11.1/scipy/_lib/_finite_differences.py
            g0.append(derivative(f, p0[i], dx, args=(np.array(p0), i)))
        g0 = np.array(g0)

        self.assertAlmostEqual(g0[0], grad[0], 3)
        self.assertAlmostEqual(g0[1], grad[1], 3)
