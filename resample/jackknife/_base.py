"""
Jackknife resampling.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from itertools import combinations
from math import comb
import random

import numpy as np

from .. import Resampler
from ._utils import (check_if_fitted, check_if_delete_1, validate_J, validate_seed,
                     validate_delete_d_input, validate_simple_block_input)

# TODO: jacknife histogram from the book  "The jackknfie and bootstrap" by Shao and Tu
# TODO: moving block jackknife

class Jackknife(Resampler):
    pass


class DeleteDJackknife(Jackknife):
    """Delete-d jackknife resampling.

    Generates jackknife samples which are subsets of size n - d.
    Computes jackknife replications.
    Estimates bias, variance and standard error of a point estimator.

    Parameters
    ----------
    sample : array-like, default = None
        one-dimensional array of observations.
    estimator : function, default = None
        point estimator - function that takes a sample as input and returns point estimate.
    d : int, default = 1
        number of observations deleted at a time to generate a jackknife sample.
    J : NoneType or int, default = None
        if None, J is set to nCd.
        Number of all possible jackknife samples is nCd.
        When nCd is large, jackknife resampling can be computationally expensive.
        In such cases, J can be set to any integer from 1 to nCd.
        J distinct jackknife samples are generated at random.
    seed : NoneType or int, default = None
        seed for random number generation if 1 <= J < nCd as J distinct jackknife samples are generated at random.
        seed must be None if J = None or J = nCd as all possible jackknife samples are generated.

    Attributes
    ----------
    n : int
        number of observations in a sample.
    nCd : int
        number of all possible delete-d jackknife samples of size n-d.
    estimate : float
        point estimate using a sample.
    replications : numpy.ndarray
        array of J jackknife replications.
    bias : float
        bias of the point estimator.
    var : float
        variance of the point estimator.
    std : float
        standard error of the point estimator.
    bc_estimate : float
        bias corrected estimate.
    influence_values : generator
        jackknife influence values.
    pseudo_values : generator
        jackknife pseudo values.

    Raises
    ------
    SampleShapeError
        if sample is not one-dimensional.
    TypeError
        if estimate_func is not callable.
        if d is not of type int.

    Examples
    --------
    >>> import numpy as np
    >>> from resample.jackknife import DeleteDJackknife
    >>> sample = np.array([10,27,31,40,46,50,52,104,146])
    >>> def estimator(sample):
    ...     return sample.mean()
    ...
    >>> ddjack = DeleteDJackknife(sample = sample, estimator = estimator)
    >>> ddjack.fit()
    >>> ddjack.bias
    0.0
    >>> ddjack.var
    199.91049382716048
    >>> ddjack.std
    14.138970748507845
    >>> ddjack.bc_estimate
    56.22222222222222
    >>> list(ddjack.influence_values)
    [-11.222222222222221, -5.972222222222221, 0.5277777777777786, 0.7777777777777786, 1.2777777777777786, 2.0277777777777786, 3.1527777777777786, 3.6527777777777786, 5.777777777777779]
    >>> list(ddjack.pseudo_values)
    [146.0, 104.0, 52.0, 50.0, 46.0, 40.0, 31.0, 27.0, 10.0]
    """

    def __init__(self, *, sample = None, estimator = None, d = 1, J = None, seed = None):

        sample = np.asarray(sample)

        # validate the inputs before instance initialization.
        validate_delete_d_input(sample = sample, estimator = estimator, d = d)

        self.sample = sample # array of observations.
        self.n = self.sample.shape[0] # number of observations in a sample.
        self.estimator = estimator # estimator function to compute point estimate.
        self.d = d # number of observations deleted at a time.
        self.nCd = comb(self.n, self.d) # number of all possible jackknife samples.

        validate_J(J = J, nCd = self.nCd) # validate type and value of J.
        self.J = self.nCd if J is None else J

        validate_seed(seed = seed, J = self.J, nCd = self.nCd) # validate the seed.
        self.seed = seed # seed for random number generation.

    def resamples(self):
        """
        Generate jackknife samples.

        Yields
        ------
        jack_sample : numpy.ndarray
            jackknife sample.
        """

        # all possible combinations of size n-d of the indices from 0 to n-1.
        # each combination is a tuple of indices of size n-d.
        c = combinations(iterable = range(self.n),r = self.n - self.d)

        # if J < nCd, generate a random sample of size J containing distinct numbers from 0 to nCd-1.
        # indices corresponding to these J numbers will be used for jackknife samples.
        if self.J < self.nCd:
            comb_indices = set(random.sample(population = range(self.nCd), k = self.J))

        # iterate over all possible combinations of indices.
        for i, indices in enumerate(c):
            if (self.J == self.nCd) or (i in comb_indices):
                jack_sample = self.sample[list(indices)] # jackknife sample of size n-d.
                yield jack_sample # yield the jackknife sample.

    def replicate(self):
        """
        Compute jackknife replications.

        Returns
        -------
        None
        """

        self.replications = np.zeros(self.J, dtype=float)  # array of size J to store jackknife replications.

        # iterate over J jackknife samples.
        for index,jack_sample in enumerate(self.resamples()):
            self.replications[index] = self.estimator(jack_sample) # compute a jackknife replication.

    def fit(self):
        """
        Compute point estimate using the given sample.
        Compute J jackknife replications.

        Returns
        -------
        None
        """

        self.estimate = self.estimator(self.sample) # point estimate using the sample.
        self.replicate() # compute jackknife replications.

    @property
    def bc_estimate(self):
        """
        Return the bias corrected estimate of a point estimator.

        Returns
        -------
        result : float
            bias corrected estimate of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        NotComputableError
            if d != 1.
        """

        check_if_fitted(self) # check that the instance is already fitted.

        check_if_delete_1(self) # check that d = 1.

        result = self.estimate - self.bias # compute bias corrected estimate.
        return result # return the bias corrected estimate.

    @property
    def influence_values(self):
        """
        Yield the jackknife influence values (aka jackknife deviations).

        Yields
        ------
        influence_value : float
            jackknife influence value.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self) # check that the instance is already fitted.

        # iterate over J jackknife replications.
        for replication in self.replications:
            influence_value = replication - self.estimate # compute influence value.
            yield influence_value # yield influence value.

    @property
    def pseudo_values(self):
        """
        Yield the pseudo values.

        Yields
        ------
        pseudo_value : float
            jackknife pseudo value.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        NotComputableError
            if d != 1.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        check_if_delete_1(self) # check that d = 1.

        # iterate over n jackknife replications.
        for replication in self.replications:
            pseudo_value = self.n * self.estimate - (self.n - 1) * replication # compute pseudo value.
            yield pseudo_value # yield pseudo value.

    @property
    def bias(self):
        """
        Estimate and return the bias of a point estimator.

        Returns
        -------
        result : float
            bias of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        NotComputableError
            if d != 1.
        """

        check_if_fitted(self) # check that the instance is already fitted.

        check_if_delete_1(self) # check that d = 1.

        result = (self.n - 1) * (self.replications.mean() - self.estimate) # compute bias of a point estimator.
        return result # return the bias.

    @property
    def var(self):
        """
        Estimate and return the variance of a point estimator.

        Returns
        -------
        result : float
            variance of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self) # check that the instance is already fitted.

        result = ((self.n - self.d) / (self.d * self.nCd)) * np.sum(np.power(self.replications - self.replications.mean(),2))
        return result # return the variance.

    @property
    def std(self):
        """
        Estimate and return the standard error of a point estimator.

        Returns
        -------
        result : float
            standard error of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self) # check that the instance is already fitted.

        result = (self.var)**0.5 # compute standard error of a point estimator.
        return result # return the standard error.

class SimpleBlockJackknife(Jackknife):

    """Simple block jackknife resampling.

    Divides a sample of size n into non-overlapping blocks of size h.
    If g = n/h is not an integer, then g = floor(n/h). Uses first g.h observations.
    Generates jackknife samples of size h(g-1) by deleting a block at a time.
    Computes jackknife replications.
    Estimates bias, variance and standard error of a point estimator.

    Parameters
    ----------
    sample : array-like, default = None
        one-dimensional array of observations.
    estimator : function, default = None
        point estimator - function that takes a sample as input and returns point estimate.
    h : int, default = 1
        number of observations in a block.

    Attributes
    ----------
    n : int
        number of observations in a sample.
    g : int
        number of simple blocks.
    gh : int
        number of observations used from a sample. (g * h)
    estimate : float
        point estimate using a sample.
    replications : numpy.ndarray
        array of g jackknife replications.
    bias : float
        bias of the point estimator.
    var : float
        variance of the point estimator.
    std : float
        standard error of the point estimator.
    bc_estimate : float
        bias corrected estimate.
    influence_values : generator
        jackknife influence values.
    pseudo_values : generator
        jackknife pseudo values.

    Raises
    ------
    SampleShapeError
        if sample is not one-dimensional.
    TypeError
        if estimate_func is not callable.
        if d is not of type int.

    Examples
    --------
    >>> import numpy as np
    >>> from resample.jackknife import SimpleBlockJackknife
    >>> sample = np.array([10,27,31,40,46,50,52,104,146], dtype = float)
    >>> def estimator(sample):
    ...     return sample.mean()
    ...
    >>> sbjack = SimpleBlockJackknife(sample = sample, estimator = estimator, h = 3)
    >>> sbjack.fit()
    >>> sbjack.bias
    0.0
    >>> sbjack.var
    536.641975308642
    >>> sbjack.std
    23.165534211596373
    >>> sbjack.bc_estimate
    56.22222222222222
    >>> list(sbjack.influence_values)
    [16.77777777777778, 5.444444444444443, -22.22222222222222]
    >>> list(sbjack.pseudo_values)
    [22.666666666666657, 45.33333333333333, 100.66666666666666]
    """

    def __init__(self, *, sample = None, estimator = None, h = 1):

        sample = np.asarray(sample)

        # validate the inputs before instance initialization.
        validate_simple_block_input(sample = sample, estimator = estimator, h = h)

        self.sample = sample # array of observations.
        self.n = self.sample.shape[0] # number of observations in a sample.
        self.estimator = estimator # estimator function to compute point estimate.
        self.h = h # number of observations in a block.
        self.g = self.n // self.h # number of nonoverlapping blocks.
        self.gh = self.g * self.h # uses first g.h observations.

    def resamples(self):
        """
        Generate jackknife samples.

        Yields
        ------
        jack_sample : numpy.ndarray
            jackknife sample.
        """

        for i in range(self.g):
            indices = list(range(0,self.h * i)) + list(range(self.h * (i+1),self.gh))
            jack_sample = self.sample[indices]
            yield jack_sample

    def replicate(self):
        """
        Compute jackknife replications.

        Returns
        -------
        None
        """

        self.replications = np.zeros(self.g, dtype = float)

        for index,jack_sample in enumerate(self.resamples()):
            self.replications[index] = self.estimator(jack_sample)  # compute a jackknife replication.

    def fit(self):
        """
        Compute point estimate using the given sample.
        Compute J jackknife replications.

        Returns
        -------
        None
        """

        self.estimate = self.estimator(self.sample)  # point estimate using the sample (all observations).
        self.replicate()  # compute jackknife replications.

    @property
    def bc_estimate(self):
        """
        Return the bias corrected estimate of a point estimator.

        Returns
        -------
        result : float
            bias corrected estimate of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        result = self.estimate - self.bias  # compute bias corrected estimate.
        return result  # return the bias corrected estimate.

    @property
    def influence_values(self):
        """
        Yield the jackknife influence values (aka jackknife deviations).

        Yields
        ------
        influence_value : float
            jackknife influence value.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        # iterate over g jackknife replications.
        for replication in self.replications:
            influence_value = replication - self.estimate  # compute influence value.
            yield influence_value  # yield influence value.

    @property
    def pseudo_values(self):
        """
        Yield the pseudo values.

        Yields
        ------
        pseudo_value : float
            jackknife pseudo value.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        # iterate over g jackknife replications.
        for replication in self.replications:
            pseudo_value = self.g * self.estimate - (self.g - 1) * replication  # compute pseudo value.
            yield pseudo_value  # yield pseudo value.

    @property
    def bias(self):
        """
        Estimate and return the bias of a point estimator.

        Returns
        -------
        result : float
            bias of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        result = (self.g - 1) * (self.replications.mean() - self.estimate)  # compute bias of a point estimator.
        return result  # return the bias.

    @property
    def var(self):
        """
        Estimate and return the variance of a point estimator.

        Returns
        -------
        result : float
            variance of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        result = ((self.g - 1) / self.g) * np.sum(np.power(self.replications - self.replications.mean(), 2))
        return result  # return the variance.

    @property
    def std(self):
        """
        Estimate and return the standard error of a point estimator.

        Returns
        -------
        result : float
            standard error of a point estimator.

        Raises
        ------
        NotFittedError
            if self.fit() is not yet called.
        """

        check_if_fitted(self)  # check that the instance is already fitted.

        result = (self.var) ** 0.5  # compute standard error of a point estimator.
        return result  # return the standard error.

class MovingBlockJackknife(Jackknife):
    pass