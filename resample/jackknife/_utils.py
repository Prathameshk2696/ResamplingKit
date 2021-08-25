"""
This module contains utility functions for the jackknife module.
"""

# Author: Prathamesh Kulkarni <prathamesh.kulkarni@rutgers.edu>

from ..exceptions import SampleShapeError, NotFittedError, NotComputableError

def check_if_fitted(jack):
    """
    Check if the jackknife is fitted.

    Parameters
    ----------
    jack: resample.jackknife.Jackknife
        instance of Jackknife.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        if jack.fit() is not yet called.
    """

    jack_attributes = {'estimate', 'replications'} # attributes that jack must have if it is fitted.

    # if any attribute in jack_attributes is not an attribute of jack
    if not (jack_attributes <= jack.__dict__.keys()):
        message = ('\n\tJackknife is not yet fitted.' +
                    "\n\tCall 'fit' before using this resampler.\n")
        raise NotFittedError(message)

def check_if_delete_1(jack):
    """
    Check if jack.d == 1.

    Returns
    -------
    None

    Raises
    ------
    NotComputableError
        if jack.d != 1
    """

    if jack.d != 1:
        message = '\n\tIt can computed only for delete-1 jackknife.\n'
        raise NotComputableError(message)

def validate_sample(*, sample):
    """
    Validate that the sample is one-dimensional.

    Parameters
    ----------
    sample : numpy.ndarray
        one-dimensional array of observations.

    Returns
    -------
    None

    Raises
    ------
    SampleShapeError
        if sample is not one-dimensional.
    """

    if len(sample.shape) != 1:
        message = '\n\t' + 'Jackknife accepts one-dimensional sample only.' + '\n'
        raise SampleShapeError(message)

def validate_estimator(*, estimator):
    """
    Validate that the estimator is callable.

    Parameters
    ----------
    estimator : function
        point estimator - function that takes a sample as input and returns point estimate.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        if estimator is not callable.
    """

    if not callable(estimator):
        message = '\n\t' + 'estimate_func must be callable.' + '\n'
        raise TypeError(message)

def validate_d(*, d, sample):
    """
    Validate the type and value of d.

    Parameters
    ----------
    d : int
        number of observations deleted at a time to generate a jackknife sample.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        if d is not of type int.
    ValueError
        if d < 1 or d >= sample.size
    """

    if not isinstance(d, int):
        message = '\n\t' + 'd must be of type int.' + '\n'
        raise TypeError(message)

    if d < 1 or d >= sample.size:
        message = '\n\t' + 'd must be a positive integer less than the number of observations in a sample.' + '\n'
        raise ValueError(message)

def validate_J(*, J, nCd):
    """
    Validate the J given as input to jackknife.

    Parameters
    ----------
    J : NoneType or int
        if None, J is set to nCd.
        Number of all possible jackknife samples is nCd.
        When nCd is large, jackknife resampling can be computationally expensive.
        In such cases, J can be set to any integer from 1 to nCd.
        J distinct jackknife samples are generated at random.

    nCd : int
        number of all possible jackknife samples.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        if J is of type neither NoneType nor int.
    ValueError
        if J < 1 or J > nCd.
    """

    if not isinstance(J, (type(None),int)):
        message = '\n\t' + 'J must be of type either NoneType or int.' + '\n'
        raise TypeError(message)

    if J is not None:
        if J < 1 or J > nCd:
            message = ('\n\t' + f'Number of all possible jackknife samples is {nCd}.' +
                       '\n\t' + f'J must be a positive integer less than or equal to {nCd}.' + '\n')
            raise ValueError(message)

def validate_seed(*, seed, J, nCd):
    """
    Validate the seed given as input to jackknife.

    Parameters
    ----------
    seed : NoneType or int
        seed for random number generation if J < nCk, i.e., a random sample of replications is computed.

    Raises
    ------
    TypeError
        if J is None and seed is not of type NoneType.
        if seed is of type neither NoneType nor int.
    """

    if (J == nCd) and (not isinstance(seed, type(None))):
        message = ('\n\t' + 'seed must be of type NoneType when J = nCd, i.e.,' +
                   '\n\twhen all possible delete-d jackknife samples are generated.' + '\n')
        raise TypeError(message)

    if not isinstance(seed, (type(None), int)):
        message = '\n\t' + 'seed must be of type either NoneType or int.' + '\n'
        raise TypeError(message)

def validate_h(*, h, sample):
    """
    Validate the type and value of h.

    Parameters
    ----------
    h : int
        number of observations in a block.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        if h is not of type int.
    ValueError
        if h < 1 or h >= sample.size
    """

    if not isinstance(h, int):
        message = '\n\t' + 'h must be of type int.' + '\n'
        raise TypeError(message)

    if h < 1 or h >= sample.size:
        message = '\n\t' + 'h must be a positive integer less than the number of observations in a sample.' + '\n'
        raise ValueError(message)



def validate_delete_d_input(*, sample, estimator, d):
    """
    Validate the inputs to Jackknife.

    Parameters
    ----------
    sample : array-like
        one-dimensional array of observations.
    estimator : function
        point estimator - function that takes a sample as input and returns point estimate.
    d : int
        number of observations deleted at a time to generate a jackknife sample.

    Returns
    -------
    None

    Raises
    ------
    SampleShapeError
        if sample is not one-dimensional.
    TypeError
        if estimator is not callable.
        if d is not of type int.
    ValueError
        if d < 1 or d >= sample.size
    """

    validate_sample(sample = sample)

    validate_estimator(estimator = estimator)

    validate_d(d = d, sample = sample)

def validate_simple_block_input(*, sample, estimator, h):
    """
    Validate the inputs to SimpleBlockJackknife.

    Parameters
    ----------
    sample : array-like
        one-dimensional array of observations.
    estimator : function
        point estimator - function that takes a sample as input and returns point estimate.
    h : int
        number of observations in a block.

    Returns
    -------
    None

    Raises
    ------
    SampleShapeError
        if sample is not one-dimensional.
    TypeError
        if estimator is not callable.
        if h is not of type int.
    ValueError
        if h < 1 or h >= sample.size
    """

    validate_sample(sample = sample)

    validate_estimator(estimator = estimator)

    validate_h(h = h, sample = sample)