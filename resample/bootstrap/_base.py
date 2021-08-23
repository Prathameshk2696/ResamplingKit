"""
Bootstrap resampling.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from abc import ABCMeta, abstractmethod

import numpy as np
from tabulate import tabulate
from scipy import stats
from scipy.stats import norm

from .. import Resampler

from ..jackknife import Jackknife

from ..utils import (
    check_if_fitted,
    validate_nonparametric_bootstrap_input,
    validate_parametric_bootstrap_input,
    validate_bootstrap_ci_method,
    check_if_ci_studentized_is_computable,
)

# TODO: add confidence interval and block bootstrap.
# TODO: add semiparametric bootstrap.
# TODO: add poisson and bag of little bootstraps.
# TODO: add variance stabilization to studentized confidence interval.
# TODO: add multithreading.
# TODO: add verbosity levels.
# TODO: improved estimate of bias.

class Bootstrap(Resampler, metaclass = ABCMeta):
    """ Base class for all bootstrap resamplers."""

    def __init__(self, *samples, estimate_func = None, plugin_estimate_func = None, B0 = 100, B1 = None,
                 cl = 90, std_error = None, seed = None, verbose = False):
        if verbose:
            print(f'Initializing object of class {self.__class__.__name__}.')
        self.samples = tuple([np.asarray(sample) for sample in samples])
        self.estimate_func = estimate_func
        self.plugin_estimate_func = plugin_estimate_func
        self.B0 = B0
        self.B1 = B1
        self.cl = cl
        self.std_error = std_error
        self.seed = seed
        self.alpha = (1 - 0.01 * self.cl) / 2
        self.verbose = verbose

    def fit(self, set_seed = True):
        """
        Compute point estimate as well as plugin estimate using the given sample (all observations) and compute bootstrap replications.

        Returns
        -------
        None
        """

        if set_seed:
            np.random.seed(seed = self.seed)
        if self.verbose: print('Computing estimate using sample.')
        self.original_estimate = self.estimate_func(*self.samples)
        if self.verbose: print('Computing plugin estimate using sample.')
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

        if self.verbose: print(f'checking that the {self.__class__.__name__} instance is fitted.')
        check_if_fitted(self)
        if self.verbose: print('estimating bias.')
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

        if self.verbose: print(f'checking that the {self.__class__.__name__} instance is fitted.')
        check_if_fitted(self)
        if self.verbose: print('estimating variance.')
        result = (np.sum((self.replications - self.replications.mean()) ** 2)) / (self.B0 - 1)
        return result

    def std(self):
        """
        Estimate standard error of a point estimator.
        Returns
        -------
        result : float
            standard error of a point estimator.
        """

        if self.verbose: print(f'checking that the {self.__class__.__name__} instance is fitted.')
        check_if_fitted(self)
        if self.verbose: print('estimating standard error.')
        result = ((np.sum((self.replications - self.replications.mean()) ** 2)) / (self.B0 - 1))**0.5
        return result

    def ci(self, method = 'basic', **kwargs):
        """
        Computes the confidence interval of a point estimator.

        Parameters
        ----------
        method : str, default = basic
            'basic' for basic bootstrap / reverse percentile interval
            'percentile' for percentile interval
            'studentized' for studentized bootstrap interval
        """

        if self.verbose: print('validating method attribute.')
        validate_bootstrap_ci_method(method)

        if method == 'basic': result = self._ci_basic()
        elif method == 'percentile': result = self._ci_percentile()
        elif method == 'studentized': result = self._ci_studentized()
        elif method == 'BC': result = self._ci_BCa(accelerated = False)
        elif method == 'BCa': result = self._ci_BCa()
        elif method == 'ABC': result = self._ci_ABC()

        return result

    def _ci_basic(self):
        """
        Compute basic bootstrap / reverse percentile interval.

        Returns
        -------
        result : tuple
            basic bootstrap interval of a point estimator.
        """

        if self.verbose:
            print()
            print(f'checking that the {self.__class__.__name__} instance is fitted.')
        check_if_fitted(self)

        if self.verbose: print('estimating basic bootstrap confidence interval.')

        if (self.B0 * self.alpha).is_integer():
            k = int(self.B0 * self.alpha)
            estimate_lo = 2 * self.original_estimate - self.replications[self.B0 - k - 1]
            estimate_up = 2 * self.original_estimate - self.replications[k - 1]
        else:
            k = int((self.B0 + 1) * self.alpha)
            estimate_lo = 2 * self.original_estimate - self.replications[self.B0 - k]
            estimate_up = 2 * self.original_estimate - self.replications[k - 1]

        result = (estimate_lo, estimate_up)
        return result

    def _ci_percentile(self):
        """
        Compute percentile interval.

        Returns
        -------
        result : tuple
            percentile interval of a point estimator.
        """

        if self.verbose:
            print()
            print(f'checking that the {self.__class__.__name__} instance is fitted.')
        check_if_fitted(self)

        if self.verbose: print('estimating percentile bootstrap confidence interval.')

        if (self.B0 * self.alpha).is_integer():
            k = int(self.B0 * self.alpha)
            estimate_lo = self.replications[k - 1]
            estimate_up = self.replications[self.B0 - k - 1]
        else:
            k = int((self.B0 + 1) * self.alpha)
            estimate_lo = self.replications[k - 1]
            estimate_up = self.replications[self.B0 - k]

        result = (estimate_lo, estimate_up)
        return result

    def _ci_studentized(self, std_error = None):
        """
        Compute studentized bootstrap / bootstrap-t confidence interval.

        Returns
        -------
        result : tuple
            studentized bootstrap confidence interval of a point estimator.
        """

        if self.verbose:
            print()
            print(f'checking that the {self.__class__.__name__} instance is fitted.')
        check_if_fitted(self)

        check_if_ci_studentized_is_computable(self)

        std = self.std()

        if self.verbose: print('estimating percentile bootstrap confidence interval.')

        if (self.B0 * self.alpha).is_integer():
            k = int(self.B0 * self.alpha)
            t_alpha = self.studentized_replications[k - 1]
            t_1_alpha = self.studentized_replications[self.B0 - k - 1]
        else:
            k = int((self.B0 + 1) * self.alpha)
            t_alpha = self.studentized_replications[k - 1]
            t_1_alpha = self.studentized_replications[self.B0 - k]

        estimate_lo = self.original_estimate - t_1_alpha * std
        estimate_up = self.original_estimate - t_alpha * std

        result = (estimate_lo, estimate_up)
        return result


class NonparametricBootstrap(Bootstrap):
    """
    Nonparametric bootstrapping.
    NonparametricBootstrap generates nonparametric bootstrap samples, computes nonparametric bootstrap replications
    and estimates bias, variance, standard error and various confidence intervals of a point estimator.

    Parameters
    ----------
    s0, s1, ..., sn : array-like
        each array is one-dimensional sample.
    estimate_func : function
        function that takes a sample as input and returns point estimate.
    plugin_estimate_func : NoneType or function, default = None
        function that takes a sample as input and returns plugin point estimate.
        if None, estimate_func is used as plugin_estimate_func.
    B0 : int, default = 1000
        number of bootstrap replications.
    B1 : NoneType or int, default = None
        number of bootstrap replications to estimate the standard error of each of the B0 bootstrap replications for
        computing studentized bootstrap replications.
        if None, studentized confidence interval cannot be computed.
        if int, studentized confidence interval can be computed.
    cl : int or float, default = 90
        confidence level (%) of a confidence interval.
        cl must be in (0,100).
    seed : NoneType or int, default = None
        seed for numpy.random.RandomState.

    Attributes
    ----------
    original_estimate : float
        point estimate using given sample (all observations).
    plugin_estimate : float
        plugin point estimate using given sample (all observations).
    replications : numpy.ndarray
        array of bootstrap replications.
    studentized_replications : numpy.ndarray
        array of studentized bootstrap replications.
        available only if B1 is of type int.

    Examples
    --------
    >>> import numpy as np
    >>> from resample.bootstrap import NonparametricBootstrap
    >>> sample1 = np.array([94, 197, 16, 38, 99, 141, 23], dtype = float)
    >>> sample2 = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46], dtype = float)
    >>> def estimate_func(sample1, sample2):
    ...     return sample1.mean() - sample2.mean()
    ...
    >>> npboot = NonparametricBootstrap(sample1, sample2, estimate_func = estimate_func,
    ...                                 plugin_estimate_func = None, B0 = 500, B1 = 25, cl = 95)
    >>> npboot.fit()
    >>> npboot.bias2()
    0.9071428571428513
    >>> npboot.var()
    641.7126688499196
    >>> npboot.std()
    25.33204825611067
    >>> npboot.ci(method = 'basic')
    (-22.80952380952381, 84.3809523809524)
    >>> npboot.ci(method = 'percentile')
    (-23.111111111111114, 84.07936507936509)
    >>> npboot.ci(method = 'studentized')
    (-38.044815638023145, 104.89102845904694)
    """

    def __init__(self, *samples, estimate_func = None, plugin_estimate_func = None, B0 = 100, B1 = None,
                 cl = 95, std_error = None, seed = None, verbose = False):

        Bootstrap.__init__(self, *samples, estimate_func = estimate_func, plugin_estimate_func = plugin_estimate_func, B0 = B0, B1 = B1,
                           cl = cl, std_error = std_error, seed = seed, verbose = verbose)

        if self.verbose: print('Validating inputs to NonparametricBootstrap.')
        validate_nonparametric_bootstrap_input(self)

    def resamples(self):
        """
        Generate nonparametric bootstrap samples.

        Yields
        ------
        npbs : list of numpy.ndarray
            nonparametric bootstrap sample
        """

        for b in range(self.B0):
            npbs = []
            for i in range(len(self.samples)):
                indices = np.random.randint(low = 0, high = self.samples[i].shape[0], size = self.samples[i].shape[0])
                npbs.append(self.samples[i][indices])
            yield npbs

    def replicate(self):
        """
        Compute bootstrap replications.

        Returns
        -------
        None
        """

        if self.verbose:
            if self.B1 is None:
                print('Computing bootstrap replications.')
            else:
                print('Computing bootstrap replications and studentized replications.')
        self.replications = np.zeros(self.B0, dtype = float)

        if (self.B1 is not None) or (self.std_error is not None): # studentized ci
            self.studentized_replications = np.zeros(self.B0)

        for index, boot_samples in enumerate(self.resamples()):
            if self.verbose: print('Replication number :',index, end = '\r')
            replication = self.estimate_func(*boot_samples)
            self.replications[index] = replication

            if self.B1 is not None: # estimate standard error of bootstrap replication and compute studentized replications.
                boot = NonparametricBootstrap(*boot_samples, estimate_func = self.estimate_func, plugin_estimate_func = self.plugin_estimate_func,
                                              B0 = self.B1)
                boot.fit(set_seed = False)
                self.studentized_replications[index] = (replication - self.original_estimate) / boot.std()
            elif self.std_error is not None: # compute studentized replications with standard error = 1
                self.studentized_replications[index] = replication - self.original_estimate

        self.replications.sort()

        if (self.B1 is not None) or (self.std_error is not None):
            self.studentized_replications.sort()

    def _ci_BCa(self, accelerated = True):
        q = (self.replications < self.plugin_estimate).sum() / self.B0
        bc = norm.ppf(q=q)

        if accelerated:
            plugin_estimate_func = self.estimate_func if self.plugin_estimate_func is None else self.plugin_estimate_func
            jack = Jackknife(sample = self.samples[0], estimate_func = plugin_estimate_func)
            jack.fit()
            jack_diff = jack.replications.mean() - jack.replications
            a = np.sum(np.power(jack_diff, 3)) / (6 * np.power(np.sum(np.power(jack_diff, 2)), 1.5))
        else:
            a = 0

        z_alpha = norm.ppf(self.alpha)
        z_1_alpha = norm.ppf(1 - self.alpha) # change to -z_alpha
        alpha_1 = norm.cdf(bc + ((bc + z_alpha)/(1 - a * (bc + z_alpha))))
        alpha_2 = norm.cdf(bc + ((bc + z_1_alpha) / (1 - a * (bc + z_1_alpha))))

        if (self.B0 * alpha_1).is_integer():
            k = int(self.B0 * alpha_1)
            estimate_lo = self.replications[k - 1]
        else:
            k = int((self.B0 + 1) * alpha_1)
            estimate_lo = self.replications[k - 1]

        if (self.B0 * alpha_2).is_integer():
            k = int(self.B0 * alpha_2)
            estimate_up = self.replications[k - 1]
        else:
            k = int((self.B0 + 1) * alpha_2)
            estimate_up = self.replications[k - 1]

        result = (estimate_lo, estimate_up)
        return result

    def _ci_ABC(self):
        pass

    def __str__(self):
        accuracy_measures = ['Bias', 'Variance', 'Standard Error', 'Basic CI', 'Percentile CI', 'Studentized CI', 'BC CI', 'BCa CI']
        values = [self.bias(), self.var(), self.std(), self.ci(method = 'basic'), self.ci(method = 'percentile'),
                  self.ci(method = 'studentized'), self.ci(method = 'BC'), self.ci(method = 'BCa')]
        return tabulate(zip(accuracy_measures, values), tablefmt='grid')


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

    def __init__(self, *samples, estimate_func = None, plugin_estimate_func = None, B0 = 100, B1 = None,
                 std_error = None, cl = 95, dists = None, seed = None, verbose = False):
        Bootstrap.__init__(self, *samples, estimate_func = estimate_func, plugin_estimate_func = plugin_estimate_func,
                           B0 = B0, B1 = B1, std_error = std_error, cl = cl, seed = seed, verbose = verbose)
        self.dists = dists # tuple of distribution names
        if self.verbose: print('Validating inputs to ParametricBootstrap.')
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
        for b in range(self.B0):
            pbs = []
            for sample, dist_object, dist_parameters_estimate in zip(self.samples, self.dist_objects, self.dist_parameters_estimates):
                pbs.append(dist_object.rvs(*dist_parameters_estimate, size = sample.shape[0]))
            yield pbs

    def replicate(self):
        """
        Compute bootstrap replications.

        Returns
        -------
        None
        """

        self.replications = np.zeros(self.B0, dtype = float)

        if (self.B1 is not None) or (self.std_error is not None):
            self.studentized_replications = np.zeros(self.B0)

        for index, boot_samples in enumerate(self.resamples()):
            replication = self.estimate_func(*boot_samples)
            self.replications[index] = replication

            if self.B1 is not None:
                boot = ParametricBootstrap(*boot_samples, estimate_func = self.estimate_func, plugin_estimate_func = self.plugin_estimate_func,
                                              B0 = self.B1, dists = self.dists)
                boot.fit(set_seed = False)
                self.studentized_replications[index] = (replication - self.original_estimate) / boot.std()
            elif self.std_error is not None: # compute studentized replications with standard error = 1
                self.studentized_replications[index] = replication - self.original_estimate

        self.replications.sort()

        if (self.B1 is not None) or (self.std_error is not None):
            self.studentized_replications.sort()