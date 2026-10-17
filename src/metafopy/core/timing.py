"""The ``time=True`` argument metafor's fitting and diagnostic functions read through ``...``.

metafor measures the elapsed time of the call and prints ``Processing time: 0 hours, 0 minutes, 1.23 seconds`` after the result
(``.print.time``). :func:`timed` gives a function the same keyword. See
https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.r and the ``time`` entry of
https://wviechtb.github.io/metafor/reference/misc-options.html.
"""

import annotationlib
import functools
import inspect
import time as clock
import typing

__all__ = ['format_time', 'timed']


def number(x: float) -> str:
    """R's ``as.character`` of a rounded number: whole numbers without a decimal point."""
    return str(int(x)) if x == int(x) else repr(x)


def format_time(x: float) -> str:
    """The message of metafor's ``.print.time`` for ``x`` seconds, with the blank lines before and after it."""
    hours = x // 3600
    minutes = x // 60 - hours * 60
    seconds = round(x - minutes * 60 - hours * 3600, 0 if x > 60 else 2)
    h = 'hours,' if hours == 0 or hours > 1 else 'hour,'
    m = 'minutes,' if minutes == 0 or minutes > 1 else 'minute,'
    s = 'seconds' if x < 60 or seconds == 0 or seconds > 1 else 'second'
    return f'\nProcessing time: {number(hours)} {h} {number(minutes)} {m} {number(seconds)} {s}\n'


def timed[**P, R](fn: typing.Callable[P, R]) -> typing.Callable[P, R]:
    """Give ``fn`` the keyword ``time``: with ``time=True`` the processing time is printed after the call, as metafor does with ``time=TRUE``."""
    sig = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    if 'time' in sig.parameters:
        return fn
    params = list(sig.parameters.values())
    extra = inspect.Parameter('time', inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool)
    at = next((i for i, p in enumerate(params) if p.kind is inspect.Parameter.VAR_KEYWORD), len(params))
    params.insert(at, extra)

    @functools.wraps(fn)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> R:
        if not kwargs.pop('time', False):
            return fn(*args, **kwargs)
        start = clock.perf_counter()
        out = fn(*args, **kwargs)
        print(format_time(clock.perf_counter() - start), end='')
        return out

    wrapper.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    return typing.cast('typing.Callable[P, R]', wrapper)
