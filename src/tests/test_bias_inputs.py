"""Property tests for the text helpers of metafopy.bias.inputs (T2 candidates fp and print_vector)."""

import hypothesis
import hypothesis.strategies
import rstrategies

import metafopy.bias.inputs

CELL = hypothesis.strategies.text(alphabet='abcXYZ0123456789.-', min_size=1, max_size=8)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    p=rstrategies.probabilities(open_interval=False),
    places=hypothesis.strategies.integers(min_value=1, max_value=6),
    equal=hypothesis.strategies.booleans(),
    sep=hypothesis.strategies.booleans(),
    pname=hypothesis.strategies.sampled_from(['', 'p', 'p-val']),
)
def test_fp_prints_the_p_value_to_its_places(p: float, places: int, equal: bool, sep: bool, pname: str) -> None:
    """bias/inputs.py::fp: below 10**-digits it prints ``<.0..1``, else the p-value to ``digits`` decimals, behind the name and sign."""
    out = metafopy.bias.inputs.fp(p, places, equal=equal, sep=sep, pname=pname)
    assert out.startswith(pname)
    small = p < 10.0**-places
    prefix = pname + ('<' if small else '=' if equal else '') + (' ' if sep else '')
    assert out.startswith(prefix)
    body = out[len(prefix) :]
    if small:
        assert body == '.' + '0' * (places - 1) + '1'
    else:
        assert abs(float(body) - p) <= 0.5 * 10.0**-places * (1 + 1e-9)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    p=hypothesis.strategies.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    places=hypothesis.strategies.integers(min_value=1, max_value=6),
)
def test_fp_clips_outside_the_unit_interval(p: float, places: int) -> None:
    """bias/inputs.py::fp: a value outside [0, 1] prints as the nearer end."""
    assert metafopy.bias.inputs.fp(p, places) == metafopy.bias.inputs.fp(min(max(p, 0.0), 1.0), places)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    pairs=hypothesis.strategies.lists(hypothesis.strategies.tuples(CELL, CELL), min_size=1, max_size=6),
)
def test_print_vector_right_aligns_each_column(pairs: list[tuple[str, str]]) -> None:
    """bias/inputs.py::print_vector: two equally long lines, each column as wide as its longer cell, two spaces apart, ending in a space."""
    names = [n for n, _ in pairs]
    values = [v for _, v in pairs]
    out = metafopy.bias.inputs.print_vector(names, values)
    row_n, row_v = out.split('\n')
    assert len(row_n) == len(row_v)
    assert row_n.endswith(' ')
    assert row_v.endswith(' ')
    widths = [max(len(n), len(v)) for n, v in pairs]
    assert len(row_n) == sum(widths) + 2 * (len(widths) - 1) + 1
    start = 0
    for w, n, v in zip(widths, names, values, strict=True):
        assert row_n[start : start + w] == n.rjust(w)
        assert row_v[start : start + w] == v.rjust(w)
        start += w + 2
