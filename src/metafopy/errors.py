"""Exceptions raised by metafopy.

Each one except the base class `MetafopyError` also derives from the matching built-in exception, so code that catches ``ValueError``,
``TypeError``, ``RuntimeError`` or ``NotImplementedError`` keeps working.
"""

__all__ = ['ArgumentError', 'EstimationError', 'InputError', 'InternalError', 'MetafopyError', 'NotFoundError', 'NotSupportedError']


class MetafopyError(Exception):
    """Base class of the exceptions metafopy raises on purpose."""


class InputError(MetafopyError, ValueError):
    """An argument or the data has an invalid value (metafor stops with an error message in these cases)."""


class ArgumentError(MetafopyError, TypeError):
    """An argument has the wrong type or is not accepted."""


class NotFoundError(MetafopyError, KeyError):
    """A requested column, variable, dataset, study label or key does not exist."""


class NotSupportedError(MetafopyError, NotImplementedError):
    """The requested combination is not implemented (in metafor, in metafopy, or in both)."""


class EstimationError(MetafopyError, RuntimeError):
    """An estimator failed; ``rma_uni`` falls back to the next method if one was given."""


class InternalError(MetafopyError, RuntimeError):
    """A condition that cannot happen if metafopy is correct; replaces ``assert`` in library code."""
