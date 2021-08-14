"""
Jackknife resampling.

Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>
"""

from itertools import combinations

import numpy as np

# TODO: add confidence interval and delete-d jackknife functionality.
# TODO: add input validation and check on whether jackknife is fitted.

class Jackknife:
    """
    Leave-one-out jackknife resampling.

    Jackknife generates jackknife samples, computes jackknife replications and
    estimates bias, variance, standard error and confidence interval of a point estimator.

    Parameters
    ----------
    sample : array-like, default = None
        one-dimensional array of observations.

    estimate : function
        function that takes a sample as input and returns point estimate.

    Attributes
    ----------
    n : int
        number of observations in the given sample.

    original_estimate : float
        point estimate using given sample (all observations).

    replications : numpy.ndarray
        array of jackknife replications.

    Examples
    --------
    >>> import numpy as np
    >>> from resample.jackknife import Jackknife
    >>> sample = np.array([10,27,31,40,46,50,52,104,146])
    >>> def estimate(sample):
    ...     return sample.mean()
    ...
    >>> jack = Jackknife(sample = sample, estimate = estimate)
    >>> jack.fit()
    >>> jack.bias()
    0.0
    >>> jack.var()
    199.91049382716048
    >>> jack.std()
    14.138970748507845
    >>> jack.ci()

    """

    def __init__(self, *, sample = None, estimate = None):
        self.sample = np.asarray(sample)
        self.n = self.sample.shape[0]
        self.estimate = estimate

    def resamples(self):
        """
        Generate jackknife samples.

        Yields
        ------
        js : numpy.ndarray
            jackknife sample
        """
        c = (list(x) for x in combinations(range(self.n),self.n - 1))
        for indices in c:
            js = self.sample[indices]
            yield js

    def replicate(self):
        """
        Compute jackknife replications.

        Returns
        -------
        None
        """
        self.replications = np.zeros(self.n, dtype = float)
        for index,sample in enumerate(self.resamples()):
            self.replications[index] = self.estimate(sample)

    def fit(self):
        """
        Compute point estimate using the given sample (all observations) and compute jackknife replications.

        Returns
        -------
        None
        """
        self.original_estimate = self.estimate(self.sample)
        self.replicate()

    def bias(self):
        """
        Estimate bias of a point estimator.

        Returns
        -------
        result : float
            bias of a point estimator.
        """
        result = (self.n - 1) * (self.replications.mean() - self.original_estimate)
        return result

    def var(self):
        """
        Estimate variance of a point estimator.

        Returns
        -------
        result : float
            variance of a point estimator.
        """
        result = ((self.n - 1) / self.n) * np.sum((self.replications - self.replications.mean())**2)
        return result

    def std(self, d = 1):
        """
        Estimate standard error of a point estimator.

        Returns
        -------
        result : float
            standard error of a point estimator.
        """
        result = (((self.n - 1) / self.n) * np.sum((self.replications - self.replications.mean()) ** 2))**0.5
        return result

    def ci(self):
        pass
