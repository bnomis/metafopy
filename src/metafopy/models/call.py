"""The recorded call of a model fit, so that ``update()`` can refit it with changed arguments (metafor's ``getCall``).

R source: https://github.com/wviechtb/metafor/blob/master/R/update.rma.r;
reference page: https://wviechtb.github.io/metafor/reference/update.rma.html.
"""

import annotationlib
import collections.abc
import dataclasses
import functools
import inspect
import re
import typing
import warnings

import metafopy.core.timing
import metafopy.errors

__all__ = ['Call', 'records_call', 'update_formula']


@dataclasses.dataclass(frozen=True, slots=True)
class Call:
    """The fitting function and the arguments it was called with (defaults left out)."""

    function: collections.abc.Callable[..., typing.Any]
    arguments: dict[str, typing.Any]

    @property
    def name(self) -> str:
        return getattr(self.function, '__name__', repr(self.function))

    def __call__(self, **changes: typing.Any) -> typing.Any:
        return self.function(**{**self.arguments, **changes})

    def __repr__(self) -> str:
        args = ', '.join(f'{k}={short_repr(v)}' for k, v in self.arguments.items())
        return f'{self.name}({args})'


def short_repr(value: typing.Any) -> str:
    text = repr(value)
    return text if len(text) <= 40 else f'<{type(value).__name__}>'


def records_call[**P, R](fn: collections.abc.Callable[P, R]) -> collections.abc.Callable[P, R]:
    """Decorate a fitting function so that its result keeps the call in its ``call`` field.

    Raises:
        metafopy.errors.InternalError: the function was called with extra positional arguments.
    """
    sig = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # the warnings of the fit are caught and issued again one frame up, so that they point at the caller of the
        # fitting function (as their stacklevel=2 intends) rather than at this wrapper
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                res = fn(*args, **kwargs)
        finally:
            # also when the fit fails, as metafor gives its warnings before an error
            for w in caught:
                warnings.warn(w.message, stacklevel=2)
        if hasattr(res, 'call'):
            bound = sig.bind(*args, **kwargs)
            arguments: dict[str, typing.Any] = {}
            for name, value in bound.arguments.items():
                kind = sig.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    arguments.update(value)
                elif kind is inspect.Parameter.VAR_POSITIONAL:
                    if value:
                        raise metafopy.errors.InternalError(f'{fn.__name__}() takes no extra positional arguments')
                else:
                    arguments[name] = value
            object.__setattr__(res, 'call', Call(function=outer, arguments=arguments))
        return res

    # the Call keeps the function that is exported (module attribute), so that it can be pickled by reference
    outer = metafopy.core.timing.timed(wrapper)
    return outer


def split_terms(rhs: str) -> list[tuple[str, str]]:
    """The terms of a formula side as (sign, term) pairs, splitting at top-level ``+`` and ``-``."""
    terms: list[tuple[str, str]] = []
    depth = 0
    sign = '+'
    current = ''
    for ch in rhs:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if depth == 0 and ch in '+-':
            if current.strip():
                terms.append((sign, current.strip()))
            sign = ch
            current = ''
        else:
            current += ch
    if current.strip():
        terms.append((sign, current.strip()))
    return terms


def update_side(old: str, new: str) -> str:
    """One side of R's ``update.formula``: ``.`` stands for the old side; ``- term`` drops a term."""
    if not new.strip():
        return old
    kept: list[str] = []
    intercept = True

    def apply(sign: str, term: str) -> None:
        nonlocal intercept
        if term in ('1', '0'):
            intercept = (term == '1') == (sign == '+')
        elif sign == '+':
            if term not in kept:
                kept.append(term)
        elif term in kept:
            kept.remove(term)

    for sign, term in split_terms(new):
        if term == '.':
            for s, t in split_terms(old):
                apply('+' if s == sign else '-', t)
        else:
            apply(sign, term)
    if not kept:
        return '1' if intercept else '0'
    return ' + '.join(kept) + ('' if intercept else ' - 1')


def update_formula(old: str | None, new: str) -> str:
    """R's ``update.formula(old, new)`` for formula strings (``'~ . + x'``, ``'. ~ . - x'``).

    Raises:
        metafopy.errors.InputError: ``new`` is not a formula.
    """
    if '~' not in new:
        raise metafopy.errors.InputError(f"'{new}' is not a formula.")
    old = '~ 1' if old is None else old
    old_lhs, old_rhs = (part.strip() for part in old.split('~', 1))
    new_lhs, new_rhs = (part.strip() for part in new.split('~', 1))
    lhs = old_lhs if new_lhs in ('', '.') else re.sub(r'(?<![\w.])\.(?![\w.])', old_lhs, new_lhs)
    rhs = update_side(old_rhs, new_rhs)
    return f'{lhs} ~ {rhs}' if lhs else f'~ {rhs}'
