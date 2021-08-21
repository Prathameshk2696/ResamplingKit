"""
This module contains utility functions for the resample module.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from .exceptions import (
    NotFittedError,
    SampleShapeError,
    LengthMismatchError,
    NotComputableError,
)

from scipy import stats
from scipy.stats import rv_continuous

def check_if_fitted(resampler):
    """
    Checks if the resampler is fitted.
    raises NotFittedError if resampler does not have attribute 'replications'.

    Parameters
    ----------
    resampler: resample.Resampler instance
        instance of Jackknife, NonparametricBootstrap or ParametricBootstrap.

    Returns
    -------
    None

    """

    if 'replications' not in resampler.__dict__:
        message = '''
        {} is not yet fitted.
        Call 'fit' before using this resampler.
        '''.format(resampler.__class__.__name__)
        raise NotFittedError(message)

def validate_jackknife_input(jack):
    """
    Validates the inputs given to Jackknife.
    raises SampleShapeError if jack.sample is not one-dimensional.
    raises TypeError if jack.estimate_func is not a callable.
    """

    if len(jack.sample.shape) != 1:
        message = 'Jackknife accepts one-dimensional sample only.'
        raise SampleShapeError(message)

    if not callable(jack.estimate_func):
        message = 'estimate_func must be callable.'
        raise TypeError(message)

def validate_bootstrap_input(boot):
    """
    Validates the inputs given to Bootstrap.
    raises SampleShapeError if boot.sample is not one-dimensional.
    raises TypeError if boot.estimate_func is not a callable.
    raises TypeError if boot.plugin_estimate_func is neither None nor callable.
    raises TypeError if boot.B0 is not of type int.
    raises TypeError if boot.B1 is of type neither NoneType nor int.
    raises TypeError if boot.cl is of type neither int nor float.
    raises TypeError if boot.seed is of type neither NoneType nor int.
    """

    for sample in boot.samples:
        if len(sample.shape) != 1:
            message = 'NonparametricBootstrap accepts one-dimensional samples only.'
            raise SampleShapeError(message)

    if not callable(boot.estimate_func):
        message = 'estimate_func must be callable.'
        raise TypeError(message)

    if (boot.plugin_estimate_func is not None) and (not callable(boot.plugin_estimate_func)):
        message = 'plugin_estimate_func must be either None or callable.'
        raise TypeError(message)

    if not isinstance(boot.B0, int):
        message = 'B0 must be of type int.'
        raise TypeError(message)

    if not isinstance(boot.B1, (type(None), int)):
        message = 'B1 must be of type either NoneType or int.'
        raise TypeError(message)

    if not isinstance(boot.cl, (int, float)):
        message = 'cl must be of type either int or float.'
        raise TypeError(message)

    if not isinstance(boot.seed, (type(None),int)):
        message = 'seed must be of type either NoneType or int.'
        raise TypeError(message)

def validate_nonparametric_bootstrap_input(npboot):
    """
    Validates the inputs given to NonparametricBootstrap.
    raises SampleShapeError if npboot.sample is not one-dimensional.
    raises TypeError if npboot.estimate_func is not a callable.
    raises TypeError if npboot.plugin_estimate_func is neither None nor callable.
    raises TypeError if boot.B0 is not of type int.
    raises TypeError if boot.B1 is of type neither NoneType nor int.
    raises TypeError if boot.cl is of type neither int nor float.
    raises TypeError if boot.seed is of type neither NoneType nor int.
    """

    validate_bootstrap_input(npboot)

def validate_parametric_bootstrap_input(pboot):
    """
    Validates the inputs given to ParametricBootstrap.
    raises SampleShapeError if pboot.sample is not one-dimensional.
    raises TypeError if pboot.estimate_func is not a callable.
    raises TypeError if pboot.plugin_estimate_func is neither None nor callable.
    raises TypeError if pboot.B0 is not of type int.
    raises TypeError if pboot.B1 is of type neither NoneType nor int.
    raises TypeError if pboot.cl is of type neither int nor float.
    raises TypeError if pboot.seed is of type neither NoneType nor int.
    raises ValueError if pboot.dists contains invalid distribution name.
    raises LengthMismatchError if number of samples is not same as the length of pboot.dists
    """

    validate_bootstrap_input(pboot)

    for dist in pboot.dists:
        if not isinstance(getattr(stats, dist), rv_continuous):
            message = f'{dist} is an invalid continuous distribution.'
            raise ValueError(message)

    if len(pboot.samples) != len(pboot.dists):
        message = 'Number of samples and number of distributions must match.'
        raise ValueError(message)

def validate_bootstrap_ci_method(method):
    """
    Validates the method argument given to ci method.
    raises ValueError if method is neither of the following - basic, percentile, studentized, BCa, ABC.
    """

    if method not in {'basic', 'percentile', 'studentized', 'BCa', 'ABC'}:
        message = '''
        {} is an invalid method.
        method must be one of the following - basic, percentile, studentized, BCa, ABC.
        '''.format(method)
        raise ValueError(message)

def check_if_ci_studentized_is_computable(boot):
    """
    Checks if studentized bootstrap confidence interval can be computed.
    raises NotComputableError if B1 is None.
    """

    if boot.B1 is None:
        message = '''
        Studentized confidence interval cannot be computed.
        B1 must be of type int in order to compute studentized confidence interval.
        '''
        raise NotComputableError(message)