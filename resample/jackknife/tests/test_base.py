# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import pytest

import numpy as np
from numpy.testing import assert_allclose

from resample.jackknife import DeleteDJackknife, SimpleBlockJackknife
from resample.exceptions import SampleShapeError, NotFittedError, NotComputableError

atol = 1e-10

class TestDelete1Jackknife:

    @pytest.fixture(scope = 'class')
    def ddjack(self):
        sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype = float)
        ddjack = DeleteDJackknife(sample = sample, estimator = np.mean, d = 1)
        ddjack.fit()
        return ddjack

    def test_estimate(self, ddjack):
        actual_estimate = ddjack.estimate
        desired_estimate = 56.22222222222222
        assert_allclose(actual=actual_estimate, desired=desired_estimate, atol = atol)

    def test_replications(self, ddjack):
        actual_replications = ddjack.replications
        desired_replications = np.array([45.0, 50.25, 56.75, 57.0, 57.5, 58.25, 59.375, 59.875, 62.0], dtype = float)
        assert_allclose(actual=actual_replications, desired=desired_replications, atol = atol)

    def test_bc_estimate(self, ddjack):
        actual_bc_estimate = ddjack.bc_estimate
        desired_bc_estimate = 56.22222222222222
        assert_allclose(actual = actual_bc_estimate, desired = desired_bc_estimate, atol = atol)

    def test_influence_values(self, ddjack):
        actual_influence_values = np.array(list(ddjack.influence_values), dtype = float)
        desired_influence_values = np.array([-11.222222222222221, -5.972222222222221, 0.5277777777777786, 0.7777777777777786, 1.2777777777777786,
                                            2.0277777777777786, 3.1527777777777786, 3.6527777777777786, 5.777777777777779], dtype = float)
        assert_allclose(actual = actual_influence_values, desired=desired_influence_values, atol = atol)

    def test_pseudo_values(self, ddjack):
        actual_pseudo_values = np.array(list(ddjack.pseudo_values), dtype=float)
        desired_pseudo_values = np.array([146.0, 104.0, 52.0, 50.0, 46.0, 40.0, 31.0, 27.0, 10.0], dtype=float)
        assert_allclose(actual=actual_pseudo_values, desired=desired_pseudo_values, atol=atol)

    def test_bias(self, ddjack):
        actual_bias = ddjack.bias
        desired_bias = 0.0
        assert_allclose(actual=actual_bias, desired=desired_bias, atol=atol)

    def test_var(self, ddjack):
        actual_variance = ddjack.var
        desired_variance = 199.91049382716048
        assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

    def test_std(self, ddjack):
        actual_std = ddjack.std
        desired_std = 14.138970748507845
        assert_allclose(actual = actual_std, desired = desired_std, atol = atol)

class TestDeleteDJackknife:

    @pytest.fixture(scope = 'class')
    def jack(self):
        sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype = float)
        jack = DeleteDJackknife(sample = sample, estimator = np.mean, d = 2)
        jack.fit()
        return jack

    def test_estimate(self, jack):
        actual_estimate = jack.estimate
        desired_estimate = 56.22222222222222
        assert_allclose(actual=actual_estimate, desired=desired_estimate, atol = atol)

    def test_replications(self, jack):
        actual_replications = jack.replications
        desired_replications = np.array([36.57142857142857, 44.0, 50.0, 44.285714285714285, 50.285714285714285, 57.714285714285715,
                                         44.857142857142854, 50.857142857142854, 58.285714285714285, 58.57142857142857, 45.714285714285715,
                                         51.714285714285715, 59.142857142857146, 59.42857142857143, 60.0, 47.0, 53.0, 60.42857142857143,
                                         60.714285714285715, 61.285714285714285, 62.142857142857146, 47.57142857142857, 53.57142857142857,
                                         61.0, 61.285714285714285, 61.857142857142854, 62.714285714285715, 64.0, 50.0, 56.0, 63.42857142857143,
                                         63.714285714285715, 64.28571428571429, 65.14285714285714, 66.42857142857143, 67.0], dtype = float)
        assert_allclose(actual=actual_replications, desired=desired_replications, atol = atol)

    def test_bc_estimate(self, jack):
        with pytest.raises(NotComputableError):
            jack.bc_estimate

    def test_influence_values(self, jack):
        actual_influence_values = np.array(list(jack.influence_values), dtype = float)
        desired_influence_values = np.array([-19.650793650793652, -12.222222222222221, -6.222222222222221, -11.936507936507937,
                                             -5.936507936507937, 1.4920634920634939, -11.365079365079367, -5.365079365079367, 2.0634920634920633,
                                             2.349206349206348, -10.507936507936506, -4.507936507936506, 2.9206349206349245, 3.206349206349209,
                                             3.7777777777777786, -9.222222222222221, -3.2222222222222214, 4.206349206349209, 4.492063492063494,
                                             5.063492063492063, 5.9206349206349245, -8.650793650793652, -2.650793650793652, 4.777777777777779,
                                             5.063492063492063, 5.634920634920633, 6.492063492063494, 7.777777777777779, -6.222222222222221,
                                             -0.22222222222222143, 7.206349206349209, 7.492063492063494, 8.06349206349207, 8.920634920634917,
                                             10.20634920634921, 10.777777777777779], dtype = float)
        assert_allclose(actual = actual_influence_values, desired=desired_influence_values, atol = atol)

    def test_pseudo_values(self, jack):
        with pytest.raises(NotComputableError):
            list(jack.pseudo_values)

    def test_bias(self, jack):
        with pytest.raises(NotComputableError):
            jack.bias

    def test_var(self, jack):
        actual_variance = jack.var
        desired_variance = 199.91049382716054
        assert_allclose(actual = actual_variance, desired = desired_variance, atol = atol)

    def test_std(self, jack):
        actual_std = jack.std
        desired_std = 14.138970748507846
        assert_allclose(actual = actual_std, desired = desired_std, atol = atol)

class TestSimpleBlockJackknife:

    @pytest.fixture(scope='class')
    def sbjack(self):
        sample = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146], dtype=float)
        sbjack = SimpleBlockJackknife(sample=sample, estimator=np.mean, h = 3)
        sbjack.fit()
        return sbjack

    def test_estimate(self, sbjack):
        actual_estimate = sbjack.estimate
        desired_estimate = 56.22222222222222
        assert_allclose(actual=actual_estimate, desired=desired_estimate, atol=atol)

    def test_replications(self, sbjack):
        actual_replications = sbjack.replications
        desired_replications = np.array([73.0, 61.666666666666664, 34.0], dtype=float)
        assert_allclose(actual=actual_replications, desired=desired_replications, atol=atol)

    def test_bc_estimate(self, sbjack):
        actual_bc_estimate = sbjack.bc_estimate
        desired_bc_estimate = 56.22222222222222
        assert_allclose(actual=actual_bc_estimate, desired=desired_bc_estimate, atol=atol)

    def test_influence_values(self, sbjack):
        actual_influence_values = np.array(list(sbjack.influence_values), dtype=float)
        desired_influence_values = np.array([16.77777777777778, 5.444444444444443, -22.22222222222222], dtype=float)
        assert_allclose(actual=actual_influence_values, desired=desired_influence_values, atol=atol)

    def test_pseudo_values(self, sbjack):
        actual_pseudo_values = np.array(list(sbjack.pseudo_values), dtype=float)
        desired_pseudo_values = np.array([22.666666666666657, 45.33333333333333, 100.66666666666666], dtype=float)
        assert_allclose(actual=actual_pseudo_values, desired=desired_pseudo_values, atol=atol)

    def test_bias(self, sbjack):
        actual_bias = sbjack.bias
        desired_bias = 0.0
        assert_allclose(actual=actual_bias, desired=desired_bias, atol=atol)

    def test_var(self, sbjack):
        actual_variance = sbjack.var
        desired_variance = 536.641975308642
        assert_allclose(actual=actual_variance, desired=desired_variance, atol=atol)

    def test_std(self, sbjack):
        actual_std = sbjack.std
        desired_std = 23.165534211596373
        assert_allclose(actual=actual_std, desired=desired_std, atol=atol)