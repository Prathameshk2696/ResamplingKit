# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import pytest

import numpy as np
from numpy.testing import assert_allclose

from resample.bootstrap import NonparametricBootstrap

# TODO: add test for bootstrap confidence interval

atol = 1e-10
random_seed = 42

def test_nonparametric_bootstrap_bias():
    np.random.seed(42)
    treatment_sample = np.array([94, 197, 16, 38, 99, 141, 23], dtype = float)
    control_sample = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
    def estimate_func(treatment_sample, control_sample):
        return treatment_sample.mean() - control_sample.mean()
    npboot = NonparametricBootstrap(treatment_sample, control_sample, estimate_func = estimate_func, B = 500)
    npboot.fit()
    actual_bias = npboot.bias()
    desired_bias = -1.0437142857142945
    assert_allclose(actual = actual_bias, desired = desired_bias, atol = atol)

def test_nonparametric_bootstrap_var():
    np.random.seed(random_seed)
    treatment_sample = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    control_sample = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(treatment_sample, control_sample):
        return treatment_sample.mean() - control_sample.mean()
    npboot = NonparametricBootstrap(treatment_sample, control_sample, estimate_func = estimate_func, B = 500)
    npboot.fit()
    actual_variance = npboot.var()
    desired_variance = 663.6485178681878
    assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

def test_nonparametric_bootstrap_std():
    np.random.seed(random_seed)
    treatment_sample = np.array([94, 197, 16, 38, 99, 141, 23], dtype=float)
    control_sample = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype=float)
    def estimate_func(treatment_sample, control_sample):
        return treatment_sample.mean() - control_sample.mean()
    npboot = NonparametricBootstrap(treatment_sample, control_sample, estimate_func = estimate_func, B = 500)
    npboot.fit()
    actual_std = npboot.std()
    desired_std = 25.761376474641022
    assert_allclose(actual = actual_std, desired = desired_std, atol = atol)



