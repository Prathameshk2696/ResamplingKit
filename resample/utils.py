"""
This module contains utility functions for the resample module.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from .exceptions import NotFittedError, SampleShapeError, LengthMismatchError
from scipy import stats
from scipy.stats import rv_continuous, rv_discrete

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
        raise NotFittedError(f"{resampler.__class__.__name__} is not yet fitted. Call 'fit' before using this resampler.")

def validate_jackknife_input(jack):
    """
    Validates the inputs given to Jackknife.
    raises SampleShapeError if jack.sample is not one-dimensional.
    raises TypeError if jack.estimate_func is not a callable.
    """

    if len(jack.sample.shape) != 1:
        raise SampleShapeError('Jackknife accepts one-dimensional sample only.')

    if not callable(jack.estimate_func):
        raise TypeError('estimate_func must be callable.')

def validate_bootstrap_input(boot):
    """
    Validates the inputs given to Bootstrap.
    raises SampleShapeError if boot.sample is not one-dimensional.
    raises TypeError if boot.estimate_func is not a callable.
    raises TypeError if boot.plugin_estimate_func is neither None nor callable.
    raises TypeError if boot.B is not of type int.
    """

    for sample in boot.samples:
        if len(sample.shape) > 1:
            raise SampleShapeError('NonparametricBootstrap accepts one-dimensional samples only.')

    if not callable(boot.estimate_func):
        raise TypeError('estimate_func must be callable.')

    if (boot.plugin_estimate_func is not None) and (not callable(boot.plugin_estimate_func)):
        raise TypeError('plugin_estimate_func must be either None or callable.')

    if not isinstance(boot.B, int):
        raise TypeError('B must be of type int')

def validate_nonparametric_bootstrap_input(npboot):
    """
    Validates the inputs given to NonparametricBootstrap.
    raises SampleShapeError if npboot.sample is not one-dimensional.
    raises TypeError if npboot.estimate_func is not a callable.
    raises TypeError if npboot.plugin_estimate_func is neither None nor callable.
    raises TypeError if npboot.B is not of type int.
    """

    validate_bootstrap_input(npboot)

def validate_parametric_bootstrap_input(pboot):
    """
    Validates the inputs given to ParametricBootstrap.
    raises SampleShapeError if pboot.sample is not one-dimensional.
    raises TypeError if pboot.estimate_func is not a callable.
    raises TypeError if pboot.plugin_estimate_func is neither None nor callable.
    raises TypeError if pboot.B is not of type int.
    raises ValueError if pboot.dists contains invalid distribution name.
    raises LengthMismatchError if number of samples is not same as the length of pboot.dists
    """

    validate_bootstrap_input(pboot)

    for dist in pboot.dists:
        if not isinstance(getattr(stats, dist), rv_continuous):
            raise ValueError(f'{dist} is an invalid continuous distribution.')

    if len(pboot.samples) != len(pboot.dists):
        raise ValueError('Number of samples and number of distributions must match.')