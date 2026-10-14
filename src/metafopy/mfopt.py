"""Package options: ``setmfopt`` and ``getmfopt``, ports of metafor's ``R/mfopt.r``.

metafor keeps its options in ``options("metafor")``; metafopy keeps them in the
module-level :data:`OPTIONS` store. Options and whether they have an effect in
metafopy:

- ``digits``: honoured. A scalar, or a mapping with any of the keys ``est``,
  ``se``, ``test``, ``pval``, ``ci``, ``var``, ``sevar``, ``fit``, ``het``,
  sets the number of decimals the fitted-model printouts use when ``digits`` is
  not passed to ``summary()``. metafopy's printouts use one number of decimals,
  so a mapping is reduced to its ``est`` element (or its first element).
- ``space``: honoured by the fitted-model printouts (``rma_uni``, ``rma_mv``,
  location-scale, ``selmodel``, ``rma_mh``, ``rma_peto`` and ``rma_glmm``
  results): ``False`` drops the blank line printed before and after the output.
- ``check``, ``silent``, ``style``, ``theme``, ``fg``, ``bg``, ``footsym``:
  stored and validated as metafor validates them, but they have no effect
  (metafopy prints no ANSI styling, does no version check and prints no
  start-up message).

R source: https://github.com/wviechtb/metafor/blob/master/R/mfopt.r;
reference page: https://wviechtb.github.io/metafor/reference/mfopt.html.
"""

import collections.abc
import typing

import metafopy.errors

__all__ = ['DIGIT_KEYS', 'OPTIONS', 'getmfopt', 'option_digits', 'option_space', 'setmfopt']

# auditme: accept[non-alphabetical-literal] the order of metafor's digits vector
DIGIT_KEYS = ('est', 'se', 'test', 'pval', 'ci', 'var', 'sevar', 'fit', 'het')

THEMES = ('auto', 'auto2', 'custom', 'custom2', 'dark', 'dark2', 'default', 'default2', 'light', 'light2')

# auditme: accept[mutable-module-level-exported] metafor's options are one process-wide store, set with setmfopt()
OPTIONS: dict[str, typing.Any] = {'check': True, 'silent': False, 'space': True, 'theme': 'default'}


def setmfopt(**kwargs: typing.Any) -> None:
    """Set package options, as metafor's ``setmfopt(...)``; an option set to ``None`` is removed.

    Raises:
        metafopy.errors.InputError: an option has a value of the wrong type.
    """
    for opt, value in kwargs.items():
        if value is not None:
            check_option(opt, value)
        if value is None:
            OPTIONS.pop(opt, None)
        else:
            OPTIONS[opt] = value


def check_option(opt: str, value: typing.Any) -> None:
    if opt == 'space' and not isinstance(value, bool):
        raise metafopy.errors.InputError("'space' must be a logical.")
    if opt == 'digits' and not is_numeric_vector(value):
        raise metafopy.errors.InputError("'digits' must be a numeric vector.")
    if opt == 'style' and not isinstance(value, bool | collections.abc.Mapping):
        raise metafopy.errors.InputError("'style' must be a list.")
    if opt == 'theme' and value not in THEMES:
        raise metafopy.errors.InputError("'theme' must be either 'default(2)', 'light(2)', 'dark(2)', 'auto(2)', or 'custom(2)'.")
    if opt in ('fg', 'bg') and not isinstance(value, str):
        raise metafopy.errors.InputError(f"'{opt}' must be a character string.")


def is_numeric_vector(value: typing.Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, collections.abc.Mapping):
        return all(isinstance(v, int | float) and not isinstance(v, bool) for v in value.values())
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        return all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    return False


def getmfopt(x: str | None = None, default: typing.Any = None) -> typing.Any:
    """Get one option (``x``) or, with ``x=None``, a copy of all options, as metafor's ``getmfopt(x, default=NULL)``."""
    if x is None:
        return dict(OPTIONS)
    value = OPTIONS.get(x)
    return default if value is None else value


def option_digits(digits: int | None, stored: int = 4) -> int:
    """The number of decimals a printout uses: an explicit ``digits``, else the ``digits`` option, else ``stored``."""
    if digits is not None:
        return digits
    opt = OPTIONS.get('digits')
    if opt is None:
        return stored
    if isinstance(opt, collections.abc.Mapping):
        value = opt.get('est', next(iter(opt.values()), stored))
    elif isinstance(opt, collections.abc.Sequence):
        value = opt[0] if opt else stored
    else:
        value = opt
    return int(value)


def option_space(text: str) -> str:
    """Drop the leading and trailing blank line of a printout when the ``space`` option is ``False``."""
    if OPTIONS.get('space', True) is not False:
        return text
    if text.startswith('\n'):
        text = text[1:]
    if text.endswith('\n'):
        text = text[:-1]
    return text
