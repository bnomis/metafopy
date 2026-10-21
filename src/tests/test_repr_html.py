"""Results display as HTML in Jupyter (``_repr_html_``)."""

import html

import metafopy


def make_fit() -> metafopy.RMAResult:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    return metafopy.rma(yi='yi', vi='vi', data=dat)


def test_rma_result_is_a_table() -> None:
    shown = make_fit()._repr_html_()  # auditme: accept the name is fixed by IPython's display protocol
    assert shown.startswith('<div><b>Random-Effects Model (k = 13; tau^2 estimator: REML)</b><table')
    assert '<td>-0.7145</td>' in shown
    assert 'τ² = 0.3132; I² = 92.22%' in shown
    assert 'Q(df = 12) = 152.23, p-val &lt;.0001' in shown


def test_other_results_are_their_text_in_a_pre_block() -> None:
    res = metafopy.regtest(make_fit())
    shown = res._repr_html_()  # auditme: accept the name is fixed by IPython's display protocol
    assert shown == f'<pre>{html.escape(str(res))}</pre>'
    assert 'Regression Test for Funnel Plot Asymmetry' in shown
