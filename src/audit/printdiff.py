"""Compare printed output of R and Python, tolerating the last printed digit and spacing."""

import re

import audit.config

DECIMAL = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?')
NUMBER = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')


QUOTES = str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"'})


def tokens(line: str) -> list[str]:
    return line.translate(QUOTES).replace('\t', ' ').split()


def decimals(token: str) -> int:
    mantissa = re.split('[eE]', token)[0]
    return len(mantissa.split('.')[1]) if '.' in mantissa else 0


def close_numbers(a: str, b: str) -> bool:
    """Two printed numbers with the same decimals agree to the last printed digit or to the relative difference the value comparison calls drift."""
    places = decimals(a)
    if places != decimals(b):
        return False
    x, y = float(a), float(b)
    return abs(x - y) <= 1.01 * 10.0**-places or abs(x - y) <= audit.config.DRIFT_RTOL * max(abs(x), abs(y))


def same_token(a: str, b: str) -> bool:
    if a == b:
        return True
    if NUMBER.match(a) and NUMBER.match(b):
        return close_numbers(a, b)
    # numbers inside a token (``QM(df = 2) = 15.6105,`` has the number and a comma): the rest must be the same
    if DECIMAL.search(a) and DECIMAL.search(b):
        numbers_a, numbers_b = DECIMAL.findall(a), DECIMAL.findall(b)
        return (
            len(numbers_a) == len(numbers_b)
            and DECIMAL.split(a) == DECIMAL.split(b)
            and all(x == y or close_numbers(x, y) for x, y in zip(numbers_a, numbers_b, strict=True))
        )
    return False


def same_line(a: str, b: str) -> bool:
    ta, tb = tokens(a), tokens(b)
    return len(ta) == len(tb) and all(same_token(x, y) for x, y in zip(ta, tb, strict=True))


def first_difference(r_text: str, py_text: str) -> str | None:
    """None when the two texts agree line by line, else a description of the first differing line."""
    r_lines = [ln for ln in r_text.splitlines() if ln.strip()]
    py_lines = [ln for ln in py_text.splitlines() if ln.strip()]
    for i, (a, b) in enumerate(zip(r_lines, py_lines, strict=False)):
        if not same_line(a, b):
            return f'line {i + 1}: R {a.strip()[:80]!r}, Python {b.strip()[:80]!r}'
    if len(r_lines) != len(py_lines):
        return f'R has {len(r_lines)} lines, Python {len(py_lines)}'
    return None
