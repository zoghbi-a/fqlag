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
from fqlag import cxd, psd


class TestCxd(unittest.TestCase):
    """"""

    def test__loglikelihood_derivative(self):
        """Check for a simple covariance"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3])
        p2 = np.array([0.2])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.5], p1=p1, p2=p2, dt=1.0)
        p0 = np.array([0.2, 0.2])
        logp0 = mod.loglikelihood(p0)
        logp, grad = mod.loglikelihood_derivative(p0, calc_fisher=False)
        self.assertAlmostEqual(logp, logp0)

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


    def test__derivative_with_check_grad__dt_nolog(self):
        """Check with misc.check_grad"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3])
        p2 = np.array([0.2])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.5], p1=p1, p2=p2, dt=1.0, log=False)
        p0 = np.array([0.2, 0.2])

        l, g, g0 = fqlag.misc.check_grad(mod, p0)
        self.assertAlmostEqual(g[0], g0[0], 3)
        self.assertAlmostEqual(g[1], g0[1], 3)


    def test__derivative_with_check_grad__no_dt_nolog(self):
        """Check with misc.check_grad"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3])
        p2 = np.array([0.2])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.5], p1=p1, p2=p2, dt=None, log=False)
        p0 = np.array([0.2, 0.2])

        l, g, g0 = fqlag.misc.check_grad(mod, p0)
        self.assertAlmostEqual(g[0], g0[0], 3)
        self.assertAlmostEqual(g[1], g0[1], 3)


    def test__derivative_with_check_grad__another_fql_nolog(self):
        """Check with misc.check_grad"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3, 0.2])
        p2 = np.array([0.2, 0.1])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.4, 0.5], p1=p1, p2=p2, dt=None, log=False)
        p0 = np.array([0.1, 0.1, 0.1, 0.1])

        l, g, g0 = fqlag.misc.check_grad(mod, p0)
        self.assertAlmostEqual(g[0], g0[0], 3)
        self.assertAlmostEqual(g[1], g0[1], 3)
        self.assertAlmostEqual(g[2], g0[2], 3)
        self.assertAlmostEqual(g[3], g0[3], 3)


    def test__derivative_with_check_grad__dt_log(self):
        """Check with misc.check_grad"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3])
        p2 = np.array([0.2])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.5], p1=p1, p2=p2, dt=0.1, log=True)
        p0 = np.array([0.2, 0.2])

        l, g, g0 = fqlag.misc.check_grad(mod, p0)
        self.assertAlmostEqual(g[0], g0[0], 3)
        self.assertAlmostEqual(g[1], g0[1], 3)


    def test__derivative_with_check_grad__no_dt_log(self):
        """Check with misc.check_grad"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3])
        p2 = np.array([0.2])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.5], p1=p1, p2=p2, dt=None, log=True)
        p0 = np.array([0.2, 0.2])

        l, g, g0 = fqlag.misc.check_grad(mod, p0)
        self.assertAlmostEqual(g[0], g0[0], 3)
        self.assertAlmostEqual(g[1], g0[1], 3)


    def test__derivative_with_check_grad__another_fql_log(self):
        """Check with misc.check_grad"""
        t = np.array([0.,1,2,3])
        y1 = np.array([1.,4,2,7])
        y2 = np.array([2.,3.,7,5])
        ye = np.array([0.1, 0.1, 0.1, 0.1])
        p1 = np.array([0.3, 0.2])
        p2 = np.array([0.2, 0.1])
        mod = cxd.Cxd([t,t], [y1,y2], [ye,ye], fql=[0.3, 0.4, 0.5], p1=p1, p2=p2, dt=None, log=True)
        p0 = np.array([0.1, 0.1, 0.1, 0.1])

        l, g, g0 = fqlag.misc.check_grad(mod, p0)
        self.assertAlmostEqual(g[0], g0[0], 3)
        self.assertAlmostEqual(g[1], g0[1], 3)
        self.assertAlmostEqual(g[2], g0[2], 3)
        self.assertAlmostEqual(g[3], g0[3], 3)
