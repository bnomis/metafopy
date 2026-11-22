import math

import hypothesis
import hypothesis.strategies
import rstrategies

import metafopy.printing


def test_fmt_num() -> None:
    assert metafopy.printing.fmt_num(-0.714532348) == '-0.7145'
    assert metafopy.printing.fmt_num(-0.00001) == '0.0000'
    assert metafopy.printing.fmt_num(math.nan) == 'NA'
    assert metafopy.printing.fmt_num(1.5, 2) == '1.50'


def test_fmt_pval() -> None:
    assert metafopy.printing.fmt_pval(1e-7) == '<.0001'
    assert metafopy.printing.fmt_pval(0.0123) == '0.0123'
    assert metafopy.printing.fmt_pval(0.001, digits=2) == '<.01'


def test_signif_stars() -> None:
    assert [metafopy.printing.signif_stars(p) for p in (0.0001, 0.005, 0.02, 0.07, 0.5)] == [
        '***',
        '**',
        '*',
        '.',
        '',
    ]


def test_fmt_table() -> None:
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    out = metafopy.printing.fmt_table(['estimate', 'se'], [['-0.7145', '0.1798']], row_names=['intrcpt'])
    assert out.splitlines() == ['         estimate      se', 'intrcpt   -0.7145  0.1798']


digits_values = hypothesis.strategies.integers(min_value=0, max_value=8)
cell_text = hypothesis.strategies.text(alphabet='abcdefgh0123456789.-', min_size=1, max_size=9)
name_text = hypothesis.strategies.text(alphabet='abcxyz', min_size=1, max_size=5)

Table = tuple[list[str], list[list[str]], list[str]]


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1e6, 1e6), digits=digits_values)
def test_fmt_num_is_within_half_a_unit_and_never_negative_zero(x: float, digits: int) -> None:
    """Candidate ``src/metafopy/printing.py::fmt_num``."""
    out = metafopy.printing.fmt_num(x, digits)
    assert abs(float(out) - x) <= 0.5 * 10.0**-digits * (1 + 1e-9) + 4 * math.ulp(x)
    assert not (out.startswith('-') and float(out) == 0.0)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(p=rstrategies.probabilities(open_interval=False), digits=hypothesis.strategies.integers(min_value=1, max_value=8))
def test_fmt_pval_is_the_cutoff_form_exactly_below_the_threshold(p: float, digits: int) -> None:
    """Candidate ``src/metafopy/printing.py::fmt_pval``."""
    out = metafopy.printing.fmt_pval(p, digits)
    if p < 10.0**-digits:
        assert out == '<.' + '0' * (digits - 1) + '1'
    else:
        assert out == metafopy.printing.fmt_num(p, digits)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1e-3, 1e-3), digits=digits_values, thresh=hypothesis.strategies.floats(min_value=1e-12, max_value=1e-3))
def test_zero_below_is_zero_exactly_below_the_threshold(x: float, digits: int, thresh: float) -> None:
    """Candidate ``src/metafopy/printing.py::zero_below``."""
    out = metafopy.printing.zero_below(x, digits, thresh)
    if abs(x) < thresh:
        assert out == '0'
    else:
        assert out == metafopy.printing.fmt_num(x, digits)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(p=rstrategies.probabilities(open_interval=False), q=rstrategies.probabilities(open_interval=False))
def test_signif_stars_is_monotone_with_the_cut_points(p: float, q: float) -> None:
    """Candidate ``src/metafopy/printing.py::signif_stars``."""
    lo, hi = min(p, q), max(p, q)
    assert len(metafopy.printing.signif_stars(lo)) >= len(metafopy.printing.signif_stars(hi))
    if p > 0.1:
        assert metafopy.printing.signif_stars(p) == ''
    if p <= 0.001:
        assert metafopy.printing.signif_stars(p) == '***'


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-9999.0, 9999.0))
def test_fmt_df_scalar_parses_back_and_integers_have_no_point(x: float) -> None:
    """Candidate ``src/metafopy/printing.py::fmt_df_scalar``.

    Degrees of freedom below 1e4: ``:g`` keeps six significant digits, so two decimals survive only there.
    """
    out = metafopy.printing.fmt_df_scalar(x)
    assert abs(float(out) - x) <= 0.005 + 1e-9 * abs(x)
    if x == int(x):
        assert '.' not in out


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(data=hypothesis.strategies.data(), n=hypothesis.strategies.integers(min_value=1, max_value=6))
def test_fmt_named_vector_lines_and_columns_share_one_width(data: hypothesis.strategies.DataObject, n: int) -> None:
    """Candidate ``src/metafopy/printing.py::fmt_named_vector``."""
    names = data.draw(hypothesis.strategies.lists(cell_text, min_size=n, max_size=n))
    values = data.draw(hypothesis.strategies.lists(cell_text, min_size=n, max_size=n))
    first, second = metafopy.printing.fmt_named_vector(names, values).split('\n')
    width = max(len(s) for s in [*names, *values])
    assert len(first) == len(second) == n * width + 2 * (n - 1)
    assert [first[i * (width + 2) : i * (width + 2) + width] for i in range(n)] == [name.rjust(width) for name in names]
    assert [second[i * (width + 2) : i * (width + 2) + width] for i in range(n)] == [value.rjust(width) for value in values]


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(names=hypothesis.strategies.lists(name_text, min_size=1, max_size=120))
def test_number_rows_has_equal_prefixes_and_recovers_names(names: list[str]) -> None:
    """Candidate ``src/metafopy/printing.py::number_rows``."""
    out = metafopy.printing.number_rows(names)
    assert len(out) == len(names)
    assert len({row.index(')') for row in out}) == 1
    assert [row.partition(') ')[2] for row in out] == names


@hypothesis.strategies.composite
def tables(draw: hypothesis.strategies.DrawFn) -> Table:
    ncol = draw(hypothesis.strategies.integers(min_value=1, max_value=8))
    nrow = draw(hypothesis.strategies.integers(min_value=1, max_value=5))
    header = draw(hypothesis.strategies.lists(cell_text, min_size=ncol, max_size=ncol))
    rows = [draw(hypothesis.strategies.lists(cell_text, min_size=ncol, max_size=ncol)) for _ in range(nrow)]
    names = draw(hypothesis.strategies.lists(name_text, min_size=nrow, max_size=nrow))
    return header, rows, names


def split_blocks(text: str, nrow: int) -> list[list[str]]:
    lines = text.split('\n')
    return [lines[i : i + nrow + 1] for i in range(0, len(lines), nrow + 1)]


def check_blocks(blocks: list[list[str]], header: list[str], rows: list[list[str]]) -> None:
    assert [cell for block in blocks for cell in block[0].split()] == header
    for r, row in enumerate(rows):
        assert [cell for block in blocks for cell in block[r + 1].split()[1:]] == row
    for block in blocks:
        # right-aligned cells end in the same column on every line of a block
        assert len({len(line) for line in block}) == 1 or all(line == line.rstrip() for line in block)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(table=tables(), width=hypothesis.strategies.integers(min_value=30, max_value=100))
def test_fmt_wrapped_table_keeps_every_cell_once_and_fits(table: Table, width: int) -> None:
    """Candidate ``src/metafopy/printing.py::fmt_wrapped_table``."""
    header, rows, names = table
    blocks = split_blocks(metafopy.printing.fmt_wrapped_table(header, rows, names, width), len(rows))
    check_blocks(blocks, header, rows)
    for block in blocks:
        assert all(len(line) < width for line in block) or len(block[0].split()) == 1


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(table=tables(), width=hypothesis.strategies.one_of(hypothesis.strategies.none(), hypothesis.strategies.integers(min_value=30, max_value=100)))
def test_fmt_table_keeps_every_cell_once_in_column_order(table: Table, width: int | None) -> None:
    """Candidate ``src/metafopy/printing.py::fmt_table``."""
    header, rows, names = table
    blocks = split_blocks(metafopy.printing.fmt_table(header, rows, row_names=names, width=width), len(rows))
    check_blocks(blocks, header, rows)
    if width is not None:
        assert all(len(line) <= width - 1 or len(block[0].split()) == 1 for block in blocks for line in block)
