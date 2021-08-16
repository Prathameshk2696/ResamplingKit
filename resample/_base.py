"""
Resampling.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from abc import ABCMeta, abstractmethod

import numpy as np
from tabulate import tabulate


class Resampler(metaclass = ABCMeta):
    """ Base class for all resamplers. """

    @abstractmethod
    def resamples(self):
        pass

    @abstractmethod
    def replicate(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def bias(self):
        pass

    @abstractmethod
    def var(self):
        pass

    @abstractmethod
    def std(self):
        pass

    @abstractmethod
    def ci(self):
        pass

    def __str__(self):
        accuracy_measures = ['Bias', 'Variance', 'Standard Error', 'Confidence Interval']
        values = [self.bias(), self.var(), self.std(), self.ci()]
        return tabulate(zip(accuracy_measures, values), tablefmt='grid')

    def __repr__(self):
        return self.__class__.__name__ + '()'