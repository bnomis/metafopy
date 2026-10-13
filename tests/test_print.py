from __future__ import annotations

import math

from metafopy._print import fmt_num, fmt_pval, fmt_table, signif_stars


def test_fmt_num() -> None:
    assert fmt_num(-0.714532348) == '-0.7145'
    assert fmt_num(-0.00001) == '0.0000'
    assert fmt_num(math.nan) == 'NA'
    assert fmt_num(1.5, 2) == '1.50'


def test_fmt_pval() -> None:
    assert fmt_pval(1e-7) == '<.0001'
    assert fmt_pval(0.0123) == '0.0123'
    assert fmt_pval(0.001, digits=2) == '<.01'


def test_signif_stars() -> None:
    assert [signif_stars(p) for p in (0.0001, 0.005, 0.02, 0.07, 0.5)] == [
        '***',
        '**',
        '*',
        '.',
        '',
    ]


def test_fmt_table() -> None:
    out = fmt_table(['estimate', 'se'], [['-0.7145', '0.1798']], row_names=['intrcpt'])
    assert out.splitlines() == ['         estimate      se', 'intrcpt   -0.7145  0.1798']
