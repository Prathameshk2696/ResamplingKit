# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import pytest

import numpy as np
from numpy.testing import assert_allclose

from resample.jackknife import DeleteDJackknife, SimpleBlockJackknife
from resample.exceptions import NotFittedError, NotComputableError, SampleShapeError
from resample.jackknife._utils import (
    check_if_fitted,
    check_if_delete_1,
    validate_sample,
    validate_estimator,
    validate_J,
    validate_seed,
    validate_d,
    validate_h,
    validate_delete_d_input,
    validate_simple_block_input,
)

def test_check_if_fitted():
    with pytest.raises(NotFittedError):
        sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype = float)
        ddjack = DeleteDJackknife(sample = sample, estimator = np.mean)
        check_if_fitted(ddjack)

def test_check_if_delete_1():
    with pytest.raises(NotComputableError):
        sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype=float)
        ddjack = DeleteDJackknife(sample=sample, estimator=np.mean, d = 2)
        check_if_delete_1(ddjack)

def test_validate_sample():
    with pytest.raises(SampleShapeError):
        sample = np.array([[10, 27, 31, 40, 46, 50, 52, 104, 146]], dtype=float)
        validate_sample(sample = sample)

def test_validate_estimator():
    with pytest.raises(TypeError):
        estimator = 2
        validate_estimator(estimator = estimator)

@pytest.mark.parametrize('d, error',[(2.0,TypeError),(9,ValueError)])
def test_validate_d(d, error):
    sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype = float)
    with pytest.raises(error):
        validate_d(d = d, sample = sample)

@pytest.mark.parametrize('J, error',[(5.0, TypeError),(10,ValueError)])
def test_validate_J(J, error):
    nCd = 9
    with pytest.raises(error):
        validate_J(J = J, nCd = nCd)

@pytest.mark.parametrize('seed, J, nCd',[(42,9,9),(42.0,9,9)])
def test_validate_seed(seed, J, nCd):
    with pytest.raises(TypeError):
        validate_seed(seed = seed, J = J, nCd = nCd)

@pytest.mark.parametrize('h, error',[(2.0,TypeError),(9,ValueError)])
def test_validate_h(h, error):
    sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype = float)
    with pytest.raises(error):
        validate_h(h = h, sample = sample)