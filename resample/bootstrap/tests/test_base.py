# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import pytest

import numpy as np
from numpy.testing import assert_allclose

from resample.bootstrap import NonparametricBootstrap, ParametricBootstrap

# TODO: add test for bootstrap confidence intervals - BCa and ABC.

atol = 1e-10
random_seed = 42

class TestNonparametricBootstrap:

    @pytest.fixture(scope = 'class')
    def npboot(self):
        sample = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
        def estimate_func(sample):
            return sample.mean()
        npboot = NonparametricBootstrap(sample, estimate_func = estimate_func,B0 = 1000, B1 = 25, cl = 95, seed = random_seed)
        npboot.fit()
        return npboot

    def test_bias(self, npboot):
        actual_bias = npboot.bias()
        desired_bias = -0.6008888888888961
        assert_allclose(actual = actual_bias, desired = desired_bias, atol = atol)

    def test_var(self, npboot):
        actual_variance = npboot.var()
        desired_variance = 173.91534171208247
        assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

    def test_std(self, npboot):
        actual_std = npboot.std()
        desired_std = 13.187696603731922
        assert_allclose(actual = actual_std, desired = desired_std, atol = atol)

    def test_ci_basic(self, npboot):
        actual_ci = npboot.ci(method = 'basic')
        desired_ci = (28.77777777777777, 80.44444444444444)
        assert_allclose(actual = actual_ci, desired = desired_ci)

    def test_ci_studentized(self, npboot):
        actual_ci = npboot.ci(method = 'studentized')
        desired_ci = (31.692849205061204, 128.4648433907318)
        assert_allclose(actual = actual_ci, desired = desired_ci)

    def test_ci_percentile(self, npboot):
        actual_ci = npboot.ci(method = 'percentile')
        desired_ci = (32.0, 83.66666666666667)
        assert_allclose(actual = actual_ci, desired = desired_ci)

class TestParametricBootstrap:

    @pytest.fixture(scope='class')
    def pboot(self):
        sample = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
        def estimate_func(sample):
            return sample.mean()
        pboot = ParametricBootstrap(sample, estimate_func = estimate_func, B0 = 1000, B1 = 25, cl = 95, dists = ('norm',), seed = 42)
        pboot.fit()
        return pboot

    def test_bias(self, pboot):
        actual_bias = pboot.bias()
        desired_bias = 0.0912969794016405
        assert_allclose(actual = actual_bias, desired = desired_bias, atol = atol)

    def test_var(self, pboot):
        actual_variance = pboot.var()
        desired_variance = 181.46581217398094
        assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

    def test_std(self, pboot):
        actual_std = pboot.std()
        desired_std = 13.47092469632211
        assert_allclose(actual = actual_std, desired = desired_std, atol = atol)

    def test_ci_basic(self, pboot):
        actual_ci = pboot.ci(method = 'basic')
        desired_ci = (28.905200841525172, 82.40808725987921)
        assert_allclose(actual = actual_ci, desired = desired_ci)

    def test_ci_studentized(self, pboot):
        actual_ci = pboot.ci(method = 'studentized')
        desired_ci = (19.277766301169898, 90.81767743253205)
        assert_allclose(actual = actual_ci, desired = desired_ci)

    def test_ci_percentile(self, pboot):
        actual_ci = pboot.ci(method = 'percentile')
        desired_ci = (30.036357184565233, 83.53924360291927)
        assert_allclose(actual = actual_ci, desired = desired_ci)