"""Exceptions raised by metafopy.

Each one also derives from the matching built-in exception, so code that catches ``ValueError``,
``TypeError``, ``RuntimeError`` or ``NotImplementedError`` keeps working.
"""

__all__ = ['ArgumentError', 'EstimationError', 'InputError', 'InternalError', 'MetaforError', 'NotFoundError', 'NotSupportedError']


class MetaforError(Exception):
    """Base class of every exception metafopy raises on purpose."""


class InputError(MetaforError, ValueError):
    """An argument or the data has an invalid value (metafor stops with an error message in these cases)."""


class ArgumentError(MetaforError, TypeError):
    """An argument has the wrong type or is not accepted."""


class NotFoundError(MetaforError, KeyError):
    """A requested column, dataset or key does not exist."""


class NotSupportedError(MetaforError, NotImplementedError):
    """The requested combination is not implemented (in metafor either)."""


class EstimationError(MetaforError, RuntimeError):
    """An estimator failed; ``rma_uni`` falls back to the next method if one was given."""


class InternalError(MetaforError, RuntimeError):
    """A condition that cannot happen if metafopy is correct; replaces ``assert`` in library code."""
