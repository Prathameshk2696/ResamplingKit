"""
This module contains all custom warnings and errors used across resample.
"""

class NotFittedError(AttributeError):
    pass

class SampleShapeError(ValueError):
    pass

class LengthMismatchError(ValueError):
    pass

class NotComputableError(ValueError, AttributeError):
    pass