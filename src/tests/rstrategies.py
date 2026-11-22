"""Shared hypothesis strategies for the property tests of metafopy helpers."""

import hypothesis.strategies

R_RESERVED_WORDS = frozenset(
    {
        'FALSE',
        'Inf',
        'NA',
        'NA_character_',
        'NA_complex_',
        'NA_integer_',
        'NA_real_',
        'NULL',
        'NaN',
        'TRUE',
        'break',
        'else',
        'for',
        'function',
        'if',
        'in',
        'next',
        'repeat',
        'while',
    }
)

LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'
TAIL = hypothesis.strategies.text(alphabet=LETTERS + DIGITS + '._', max_size=12)


def probabilities(*, open_interval: bool = True) -> hypothesis.strategies.SearchStrategy[float]:
    """Floats in (0, 1), or [0, 1] when open_interval is False; never NaN."""
    return hypothesis.strategies.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=open_interval,
        exclude_max=open_interval,
    )


def df_values(*, allow_inf: bool = False) -> hypothesis.strategies.SearchStrategy[float]:
    """Positive finite floats from 0.5 to 1e6, plus inf when allow_inf is True."""
    finite = hypothesis.strategies.floats(min_value=0.5, max_value=1e6, allow_nan=False, allow_infinity=False)
    if allow_inf:
        return finite | hypothesis.strategies.just(float('inf'))
    return finite


def finite_floats(lo: float = -1e12, hi: float = 1e12) -> hypothesis.strategies.SearchStrategy[float]:
    """Finite floats in [lo, hi], with no NaN or infinity."""
    return hypothesis.strategies.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


@hypothesis.strategies.composite
def identifier(draw: hypothesis.strategies.DrawFn) -> str:
    """A syntactic R name, possibly a reserved word."""
    start = draw(hypothesis.strategies.sampled_from(LETTERS + '.'))
    if start == '.':
        start += draw(hypothesis.strategies.sampled_from(LETTERS + '._'))
    return start + draw(TAIL)


def r_identifiers() -> hypothesis.strategies.SearchStrategy[str]:
    """Syntactically valid R names that are not R reserved words.

    A name starts with a letter, or a dot not followed by a digit, then letters, digits, ``.`` and ``_``.
    """
    return identifier().filter(lambda name: name not in R_RESERVED_WORDS)
