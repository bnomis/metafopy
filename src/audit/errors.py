__all__ = [
    'HarnessError',
]
"""Errors of the audit harness."""


class HarnessError(Exception):
    """A step or a run of the audit harness failed."""
