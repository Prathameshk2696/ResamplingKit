"""
Bootstrap resampling.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

import numpy as np
from tabulate import tabulate

from ..utils import check_if_fitted, validate_nonparametric_bootstrap_input

# TODO: add confidence interval and block bootstrap.

class NonparametricBootstrap:
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
        self.samples = tuple([np.asarray(sample) for sample in samples]) # one or more one-dimensional arrays
        print(self.samples)
        self.estimate_func = estimate_func
        self.plugin_estimate_func = plugin_estimate_func
        self.B = B
        validate_nonparametric_bootstrap_input(self.samples, self.estimate_func, self.plugin_estimate_func, self.B)

    def resamples(self):
        """

        """
        for b in range(self.B):
            boot_samples = []
            for i in range(len(self.samples)):
                indices = np.random.randint(low = 0, high = self.samples[i].shape[0], size = self.samples[i].shape[0])
                boot_samples.append(self.samples[i][indices])
            yield boot_samples

    def replicate(self):
        """

        """
        self.replications = np.zeros(self.B, dtype = float)
        for index, boot_samples in enumerate(self.resamples()):
            self.replications[index] = self.estimate_func(*boot_samples)

    def fit(self):
        """
        Compute point estimate using the given sample (all observations) and compute jackknife replications.

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
        check_if_fitted(self)
        pass

    def __str__(self):
        accuracy_measures = ['Bias', 'Variance', 'Standard Error', 'Confidence Interval']
        values = [self.bias(), self.var(), self.std(), self.ci()]
        return tabulate(zip(accuracy_measures,values), tablefmt = 'grid')

    def __repr__(self):
        return 'NonparametricBootstrap()'