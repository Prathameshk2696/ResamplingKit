# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import pytest

import numpy as np
from numpy.testing import assert_allclose

from resample.bootstrap import NonparametricBootstrap, ParametricBootstrap

# TODO: add test for bootstrap confidence interval

atol = 1e-10
random_seed = 42

def test_nonparametric_bootstrap_bias():
    sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype = float)
    sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
    def estimate_func(sample1, sample2):
        return sample1.mean() - sample2.mean()
    npboot = NonparametricBootstrap(sample1, sample2, estimate_func = estimate_func, B = 500, seed = random_seed)
    npboot.fit()
    actual_bias = npboot.bias()
    desired_bias = -1.0437142857142945
    assert_allclose(actual = actual_bias, desired = desired_bias, atol = atol)

def test_nonparametric_bootstrap_var():
    sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(sample1, sample2):
        return sample1.mean() - sample2.mean()
    npboot = NonparametricBootstrap(sample1, sample2, estimate_func=estimate_func, B=500, seed = random_seed)
    npboot.fit()
    actual_variance = npboot.var()
    desired_variance = 663.6485178681878
    assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

def test_nonparametric_bootstrap_std():
    sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(sample1, sample2):
        return sample1.mean() - sample2.mean()
    npboot = NonparametricBootstrap(sample1, sample2, estimate_func=estimate_func, B=500, seed = random_seed)
    npboot.fit()
    actual_std = npboot.std()
    desired_std = 25.761376474641022
    assert_allclose(actual = actual_std, desired = desired_std, atol = atol)

def test_parametric_bootstrap_bias():
    sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(sample1, sample2):
        return sample1.mean() - sample2.mean()
    pboot = ParametricBootstrap(sample1, sample2, estimate_func=estimate_func, B=200, dists = ('norm', 'norm'), seed = random_seed)
    pboot.fit()
    actual_bias = pboot.bias()
    desired_bias = 0.3561565729846521
    assert_allclose(actual = actual_bias, desired = desired_bias, atol = atol)

def test_parametric_bootstrap_var():
    sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(sample1, sample2):
        return sample1.mean() - sample2.mean()
    pboot = ParametricBootstrap(sample1, sample2, estimate_func=estimate_func, B=200, dists=('norm', 'norm'), seed = random_seed)
    pboot.fit()
    actual_variance = pboot.var()
    desired_variance = 718.0531559250003
    assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

def test_parametric_bootstrap_std():
    sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(sample1, sample2):
        return sample1.mean() - sample2.mean()
    pboot = ParametricBootstrap(sample1, sample2, estimate_func=estimate_func, B = 200, dists=('norm', 'norm'), seed = random_seed)
    pboot.fit()
    actual_std = pboot.std()
    desired_std = 26.796513876342203
    assert_allclose(actual = actual_std, desired = desired_std, atol = atol)



