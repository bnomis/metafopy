"""Compare printed output of R and Python, tolerating the last printed digit and spacing."""

import re

NUMBER = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')


QUOTES = str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"'})


def tokens(line: str) -> list[str]:
    return line.translate(QUOTES).replace('\t', ' ').split()


def decimals(token: str) -> int:
    mantissa = re.split('[eE]', token)[0]
    return len(mantissa.split('.')[1]) if '.' in mantissa else 0


def same_token(a: str, b: str) -> bool:
    if a == b:
        return True
    if NUMBER.match(a) and NUMBER.match(b):
        places = decimals(a)
        return places == decimals(b) and abs(float(a) - float(b)) <= 1.01 * 10.0**-places
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
