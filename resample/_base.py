"""
Resampling.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from abc import ABCMeta, abstractmethod

from tabulate import tabulate


class Resampler(metaclass = ABCMeta):
    """ Base class for all resamplers. """

    @abstractmethod
    def resamples(self):
        """ Yield a resample. """

    @abstractmethod
    def replicate(self):
        """ Compute replications of an estimate. """

    @abstractmethod
    def fit(self):
        """ Compute point estimate using the given sample. Compute replications. """

    @abstractmethod
    def bias(self):
        """  Return the bias of a point estimator. """

    @abstractmethod
    def var(self):
        """  Return the variance of a point estimator. """

    @abstractmethod
    def std(self):
        """  Return the standard error of a point estimator. """

    def __str__(self):
        accuracy_measures = ['Bias', 'Variance', 'Standard Error', 'Confidence Interval']
        values = [self.bias(), self.var(), self.std(), self.ci()]
        return tabulate(zip(accuracy_measures, values), tablefmt='grid')

    def __repr__(self):
        return self.__class__.__name__ + '()'