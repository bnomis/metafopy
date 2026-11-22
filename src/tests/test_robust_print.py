"""Property tests for metafopy.models.robust_print (T2 candidate decorate)."""

import hypothesis
import hypothesis.strategies

import metafopy.models.robust_print

CELL = hypothesis.strategies.text(alphabet='abc0123.-', max_size=6)


@hypothesis.strategies.composite
def tables(draw: hypothesis.strategies.DrawFn) -> tuple[list[str], list[list[str]]]:
    """A header and rows of the same number of columns."""
    ncol = draw(hypothesis.strategies.integers(min_value=1, max_value=7))
    header = draw(hypothesis.strategies.lists(CELL, min_size=ncol, max_size=ncol))
    rows = draw(hypothesis.strategies.lists(hypothesis.strategies.lists(CELL, min_size=ncol, max_size=ncol), max_size=5))
    return header, rows


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(table=tables())
def test_decorate_marks_and_pads_every_column_but_the_first(table: tuple[list[str], list[list[str]]]) -> None:
    """models/robust_print.py::decorate: the shape is kept; the footnote mark goes on each non-empty header after the first, a space after each inner cell."""
    header, rows = table
    head, body = metafopy.models.robust_print.decorate(header, rows)
    assert len(head) == len(header)
    assert len(body) == len(rows)
    for j, (old, new) in enumerate(zip(header, head, strict=True)):
        assert new == (old if j == 0 or old == '' else old + metafopy.models.robust_print.FOOT_MARK)
    for old_row, new_row in zip(rows, body, strict=True):
        assert len(new_row) == len(old_row)
        for j, (old, new) in enumerate(zip(old_row, new_row, strict=True)):
            assert new == (old if j == 0 or j == len(old_row) - 1 else old + ' ')
