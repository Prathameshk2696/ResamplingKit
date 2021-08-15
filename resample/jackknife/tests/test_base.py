# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import pytest

import numpy as np
from numpy.testing import assert_allclose

from resample.jackknife import Jackknife

# TODO: add test for jackknife confidence interval

atol = 1e-10

def test_jackknife_bias():
    sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146])
    jack = Jackknife(sample=sample, estimate=np.mean)
    jack.fit()
    actual_bias = jack.bias()
    desired_bias = 0.0
    assert_allclose(actual = actual_bias, desired = desired_bias, atol = atol)

def test_jackknife_var():
    sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146])
    jack = Jackknife(sample=sample, estimate=np.mean)
    jack.fit()
    actual_variance = jack.var()
    desired_variance = 199.91049382716048
    assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

def test_jackknife_std():
    sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146])
    jack = Jackknife(sample=sample, estimate=np.mean)
    jack.fit()
    actual_std = jack.std()
    desired_std = 14.138970748507845
    assert_allclose(actual = actual_std, desired = desired_std, atol = atol)




