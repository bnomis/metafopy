"""Package options: ``setmfopt`` and ``getmfopt``, ports of metafor's ``R/mfopt.r``.

metafor keeps its options in ``options("metafor")``; metafopy keeps them in the
module-level :data:`OPTIONS` store. Options and whether they have an effect in
metafopy:

- ``digits``: honoured. A scalar, or a mapping with any of the keys ``est``,
  ``se``, ``test``, ``pval``, ``ci``, ``var``, ``sevar``, ``fit``, ``het``,
  sets the number of decimals the fitted-model printouts use when ``digits`` is
  not passed to ``summary()``. metafopy's printouts use one number of decimals,
  so a mapping is reduced to its ``est`` element (or its first element).
- ``signif_stars``: honoured, metafopy's counterpart of R's ``getOption("show.signif.stars")``. ``False`` drops the significance stars (and,
  unless ``signif_legend`` is passed, the legend) of the coefficient tables when ``signif_stars`` is not passed to ``summary()``.
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
import dataclasses
import math
import typing

import numpy

import metafopy.errors

__all__ = [
    'DIGIT_KEYS',
    'OPTIONS',
    'digit_map_of',
    'digits_map',
    'getmfopt',
    'model_digits',
    'option_digits',
    'option_signif',
    'option_space',
    'reduce_digits',
    'setmfopt',
    'with_digits',
]

# auditme: accept[non-alphabetical-literal] the order of metafor's digits vector
DIGIT_KEYS = ('est', 'se', 'test', 'pval', 'ci', 'var', 'sevar', 'fit', 'het')

THEMES = ('auto', 'auto2', 'custom', 'custom2', 'dark', 'dark2', 'default', 'default2', 'light', 'light2')

# auditme: accept[mutable-module-level-exported] metafor's options are one process-wide store, set with setmfopt()
OPTIONS: dict[str, typing.Any] = {'check': True, 'signif_stars': True, 'silent': False, 'space': True, 'theme': 'default'}


def setmfopt(**kwargs: typing.Any) -> dict[str, dict[str, typing.Any]]:
    """Set package options, as metafor's ``setmfopt(...)``; an option set to ``None`` is removed.

    Returns the options as they were before, as ``{'metafor': options}`` (what ``options(metafor = ...)`` returns in R).

    Raises:
        metafopy.errors.InputError: an option has a value of the wrong type.
    """
    previous = dict(OPTIONS)
    for opt, value in kwargs.items():
        if value is not None:
            check_option(opt, value)
        if value is None:
            OPTIONS.pop(opt, None)
        else:
            OPTIONS[opt] = value
    return {'metafor': previous}


def check_option(opt: str, value: typing.Any) -> None:
    if opt == 'space' and not isinstance(value, bool):
        raise metafopy.errors.InputError("'space' must be a logical.")
    if opt == 'signif_stars' and not isinstance(value, bool):
        raise metafopy.errors.InputError("'signif_stars' must be a logical.")
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


def reduce_digits(value: typing.Any, stored: int = 4) -> int:
    """A ``digits`` value (a scalar, or a mapping or sequence as metafor's named vector) as the one number of decimals the printouts use:
    the ``est`` element of a mapping (else its first), the first element of a sequence."""
    if isinstance(value, collections.abc.Mapping):
        value = value.get('est', next(iter(value.values()), stored))
    elif isinstance(value, numpy.ndarray | collections.abc.Sequence) and not isinstance(value, str):
        value = value[0] if len(value) else stored
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return stored
    return int(value)


def stored_digits(digits: typing.Any, stored: int = 4) -> int:
    """The ``digits`` a result keeps: the explicit ``digits`` reduced to one number of decimals, else ``stored`` (the ``digits`` of the fit)."""
    return stored if digits is None else reduce_digits(digits, stored)


def model_digits(x: typing.Any) -> int:
    """The ``digits`` a model was fitted with (metafor's ``x$digits``), else 4 for a vector of estimates."""
    stored = getattr(x, 'digits', None)
    return stored if isinstance(stored, int) and not isinstance(stored, bool) else 4


def digits_map(digits: typing.Any = None, stored: typing.Any = 4) -> dict[str, int]:
    """The decimals of each kind of number (``est``, ``se``, ``test``, ``pval``, ``ci``, ``var``, ``sevar``, ``fit``, ``het``) a printout uses,
    metafor's ``.get.digits``: ``stored`` (a number or a mapping), changed by the ``digits`` option and then by an explicit ``digits``. A number
    or an unnamed vector sets all kinds to its first element; a mapping changes the kinds it names (a unique prefix of a name is enough).

    R: https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.r (``.get.digits``).
    """
    res = dict.fromkeys(DIGIT_KEYS, reduce_digits(stored))
    if isinstance(stored, collections.abc.Mapping):
        res.update(named_digits(stored))
    opt = OPTIONS.get('digits')
    if opt is not None:
        if isinstance(opt, collections.abc.Mapping):
            res.update(named_digits(opt))
        elif is_scalar_digits(opt):
            res = dict.fromkeys(DIGIT_KEYS, reduce_digits(opt))
    if digits is not None:
        if isinstance(digits, collections.abc.Mapping):
            res.update(named_digits(digits))
        else:
            res = dict.fromkeys(DIGIT_KEYS, reduce_digits(digits, res['est']))
    return res


def digit_map_of(digits: typing.Any, stored: typing.Any = 4) -> dict[str, int] | None:
    """:func:`digits_map` when the kinds of number differ in their decimals, else ``None`` (the one number the result stores as ``digits``)."""
    by_kind = digits_map(digits, stored)
    return None if len(set(by_kind.values())) == 1 else by_kind


def with_digits[R](result: R, digits: typing.Any, stored: typing.Any) -> R:
    """A copy of a result dataclass with the ``digits`` and ``digit_map`` of a printout made with ``digits``."""
    return dataclasses.replace(result, digits=option_digits(digits, stored), digit_map=digit_map_of(digits, stored))  # type: ignore[type-var]


def is_scalar_digits(value: typing.Any) -> bool:
    """Whether ``value`` is a single number (a vector of length one counts)."""
    return not isinstance(value, collections.abc.Mapping | str) and numpy.size(value) == 1


def named_digits(digits: collections.abc.Mapping[str, typing.Any]) -> dict[str, int]:
    """The elements of a named ``digits`` vector whose name is (the start of) one of :data:`DIGIT_KEYS`, as R's ``pmatch`` finds them."""
    out: dict[str, int] = {}
    for name, value in digits.items():
        hits = [key for key in DIGIT_KEYS if key == name] or [key for key in DIGIT_KEYS if key.startswith(str(name))]
        if len(hits) == 1 and value is not None:
            out[hits[0]] = int(value)
    return out


def option_digits(digits: typing.Any, stored: int = 4) -> int:
    """The number of decimals a printout uses (metafor's ``.get.digits``): an explicit ``digits``, else the ``digits`` option, else ``stored``
    (the ``digits`` the model was fitted with)."""
    if digits is not None:
        return reduce_digits(digits, stored)
    opt = OPTIONS.get('digits')
    if opt is None:
        return stored
    return reduce_digits(opt, stored)


def option_signif(signif_stars: bool | None, signif_legend: bool | None) -> tuple[bool, bool]:
    """``signif_stars`` (default: the ``signif_stars`` option, ``True``) and ``signif_legend`` (default: ``signif_stars``) of a printout."""
    stars = bool(OPTIONS.get('signif_stars', True)) if signif_stars is None else signif_stars
    return stars, stars if signif_legend is None else signif_legend


def option_space(text: str) -> str:
    """Drop the leading and trailing blank line of a printout when the ``space`` option is ``False``."""
    if OPTIONS.get('space', True) is not False:
        return text
    if text.startswith('\n'):
        text = text[1:]
    if text.endswith('\n'):
        text = text[:-1]
    return text
