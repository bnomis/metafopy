import math

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
