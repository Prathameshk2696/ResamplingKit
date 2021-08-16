"""
Bootstrap resampling.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from abc import ABCMeta, abstractmethod

import numpy as np
from tabulate import tabulate
from scipy import stats

from .. import Resampler

from ..utils import (
    check_if_fitted,
    validate_nonparametric_bootstrap_input,
    validate_parametric_bootstrap_input
)

# TODO: add confidence interval and block bootstrap.
# TODO: add random seed functionality.
# TODO: add semiparametric bootstrap.
# TODO: add poisson and bag of little bootstraps.

class Bootstrap(Resampler, metaclass = ABCMeta):
    """ Base class for all bootstrap resamplers."""

    def __init__(self, *samples, estimate_func = None, plugin_estimate_func = None, B = 100):
        self.samples = tuple([np.asarray(sample) for sample in samples])  # one or more one-dimensional arrays
        print(self.samples)
        self.estimate_func = estimate_func
        self.plugin_estimate_func = plugin_estimate_func
        self.B = B

    def replicate(self):
        """
        Compute bootstrap replications.

        Returns
        -------
        None
        """

        self.replications = np.zeros(self.B, dtype = float)
        for index, boot_samples in enumerate(self.resamples()):
            self.replications[index] = self.estimate_func(*boot_samples)

    def fit(self):
        """
        Compute point estimate as well as plugin estimate using the given sample (all observations) and compute bootstrap replications.

        Returns
        -------
        None
        """

        self.original_estimate = self.estimate_func(*self.samples)
        self.plugin_estimate = self.original_estimate if self.plugin_estimate_func is None else self.plugin_estimate_func(*self.samples)
        self.replicate()

    def bias(self):
        """
        Estimate bias of a point estimator.
        Returns
        -------
        result : float
            bias of a point estimator.
        """

        check_if_fitted(self)
        result = self.replications.mean() - self.plugin_estimate
        return result

    def var(self):
        """
        Estimate variance of a point estimator.
        Returns
        -------
        result : float
            variance of a point estimator.
        """

        check_if_fitted(self)
        result = (np.sum((self.replications - self.replications.mean()) ** 2)) / (self.B - 1)
        return result

    def std(self):
        """
        Estimate standard error of a point estimator.
        Returns
        -------
        result : float
            standard error of a point estimator.
        """

        check_if_fitted(self)
        result = ((np.sum((self.replications - self.replications.mean()) ** 2)) / (self.B - 1))**0.5
        return result

    def ci(self):
        """
        """

        check_if_fitted(self)
        pass


class NonparametricBootstrap(Bootstrap):
    """
    Nonparametric bootstrap resampling.
    NonparametricBootstrap generates bootstrap samples, computes bootstrap replications and
    estimates bias, variance, standard error and confidence interval of a point estimator.

    Parameters
    ----------
    s0, s1, ..., sn : array-like
        each array is one-dimensional sample.
    estimate_func : function
        function that takes a sample as input and returns point estimate.
    plugin_estimate_func : function, default = None
        function that takes a sample as input and returns plugin point estimate.
    B: int, default = 100
        number of bootstrap replications.

    Attributes
    ----------
    original_estimate : float
        point estimate using given sample (all observations).
    plugin_estimate : float
        plugin point estimate using given sample (all observations).
    replications : numpy.ndarray
        array of bootstrap replications.

    Examples
    --------
    >>> import numpy as np
    >>> from resample.bootstrap import NonparametricBootstrap
    >>> treatment_sample = np.array([94, 197, 16, 38, 99, 141, 23], dtype = float)
    >>> control_sample = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
    >>> def estimate_func(treatment_sample, control_sample):
    ...     return treatment_sample.mean() - control_sample.mean()
    ...
    >>> npboot = NonparametricBootstrap(treatment_sample, control_sample, estimate_func = estimate_func,
    ...                                 plugin_estimate_func = None, B = 500)
    >>> npboot.fit()
    >>> npboot.bias()
    0.9071428571428513
    >>> npboot.var()
    641.7126688499196
    >>> npboot.std()
    25.33204825611067
    >>> npboot.ci()

    """

    def __init__(self, *samples, estimate_func = None, plugin_estimate_func = None, B = 100):
        Bootstrap.__init__(self, samples = samples, estimate_func = estimate_func,
                         plugin_estimate_func = plugin_estimate_func, B = B)
        validate_nonparametric_bootstrap_input(self)


    def resamples(self):
        """
        Generate nonparametric bootstrap samples.

        Yields
        ------
        npbs : list of numpy.ndarray
            nonparametric bootstrap sample
        """

        for b in range(self.B):
            npbs = []
            for i in range(len(self.samples)):
                indices = np.random.randint(low = 0, high = self.samples[i].shape[0], size = self.samples[i].shape[0])
                npbs.append(self.samples[i][indices])
            yield npbs


class ParametricBootstrap(Bootstrap):
    """
    Parametric bootstrap resampling.
    ParametricBootstrap generates bootstrap samples, computes bootstrap replications and
    estimates bias, variance, standard error and confidence interval of a point estimator.

    Parameters
    ----------
    s0, s1, ..., sn : array-like
        each array is one-dimensional sample.
    estimate_func : function
        function that takes a sample as input and returns point estimate.
    plugin_estimate_func : function, default = None
        function that takes a sample as input and returns plugin point estimate.
    B: int, default = 100
        number of bootstrap replications.
    dists: tuple of str, default = None
        tuple of distribution names.

    Attributes
    ----------
    original_estimate : float
        point estimate using given sample (all observations).
    plugin_estimate : float
        plugin point estimate using given sample (all observations).
    replications : numpy.ndarray
        array of bootstrap replications.
    dist_objects : list of scipy.stats.rv_continuous instances.
        each scipy.stats.rv_continuous instance is a parametric distribution used to draw a bootstrap sample.
    dist_parameters_estimates : list of tuples
        each tuple represents maximum likelihood estimates of the parameters of corresponding scipy.stats.rv_continuous instance.

    Examples
    --------
    >>> import numpy as np
    >>> from resample.bootstrap import ParametricBootstrap
    >>> sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype = float)
    >>> sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
    >>> def estimate_func(sample1, sample2):
    ...     return sample1.mean() - sample2.mean()
    ...
    >>> pboot = ParametricBootstrap(sample1, sample2, estimate_func = estimate_func,
    ...                                 plugin_estimate_func = None, B = 500, dists = ('norm', 'norm'))
    >>> pboot.fit()
    >>> pboot.bias()
    0.9071428571428513
    >>> pboot.var()
    641.7126688499196
    >>> pboot.std()
    25.33204825611067
    >>> pboot.ci()
    """

    def __init__(self, *samples, estimate_func = None, plugin_estimate_func = None, B = 100, dists = None):
        Bootstrap.__init__(self, *samples, estimate_func = estimate_func,
                         plugin_estimate_func = plugin_estimate_func, B = B)
        self.dists = dists # tuple of distribution names
        validate_parametric_bootstrap_input(self)

    def fit_dist(self):
        """
        Compute maximum likelihood estimates of distribution parameters using given samples.

        Returns
        -------
        None
        """

        self.dist_objects = [getattr(stats, dist) for dist in self.dists]
        self.dist_parameters_estimates = []
        for sample, dist_object in zip(self.samples, self.dist_objects):
            dist_parameters_estimate = dist_object.fit(sample)
            self.dist_parameters_estimates.append(dist_parameters_estimate)

    def resamples(self):
        """
        Generate parametric bootstrap samples.

        Yields
        ------
        pbs : list of numpy.ndarray
            parametric bootstrap sample
        """

        self.fit_dist()
        for b in range(self.B):
            pbs = []
            for sample, dist_object, dist_parameters_estimate in zip(self.samples, self.dist_objects, self.dist_parameters_estimates):
                pbs.append(dist_object.rvs(*dist_parameters_estimate, size = sample.shape[0]))
            yield pbs