"""Text output of a location-scale model, after metafor's ``print.rma.uni`` (the ``rma.ls`` branch).

https://github.com/wviechtb/metafor/blob/master/R/print.rma.uni.r
"""

import math

import numpy
import numpy.typing

import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['coef_table', 'format_result']

type FloatArray = numpy.typing.NDArray[numpy.float64]


def coef_table(
    estimate: FloatArray,
    se: FloatArray,
    zval: FloatArray,
    pval: FloatArray,
    ci_lb: FloatArray,
    ci_ub: FloatArray,
    names: tuple[str, ...],
    ddf: int | None,
    fixed: FloatArray | None,
    digits: int,
    signif_stars: bool = True,
) -> str:
    """A coefficient table; the entries of fixed coefficients (``fixed``) other than the estimate print as ``---``."""
    tdist = ddf is not None
    header = ['estimate', 'se', 'tval' if tdist else 'zval']
    if tdist:
        header.append('df')
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    header += ['pval', 'ci.lb', 'ci.ub', '']
    rows: list[list[str]] = []
    for j in range(estimate.shape[0]):
        row = [
            metafopy.printing.fmt_num(float(estimate[j]), digits),
            metafopy.printing.fmt_num(float(se[j]), digits),
            metafopy.printing.fmt_num(float(zval[j]), digits),
        ]
        if tdist:
            row.append(str(ddf))
        row += [
            metafopy.printing.fmt_pval(float(pval[j]), digits),
            metafopy.printing.fmt_num(float(ci_lb[j]), digits),
            metafopy.printing.fmt_num(float(ci_ub[j]), digits),
        ]
        if fixed is not None and not math.isnan(fixed[j]):
            row = [row[0], *['---' if cell == 'NA' else cell for cell in row[1:]]]
        row.append(metafopy.printing.signif_stars(float(pval[j])))
        rows.append(row)
    header, rows = metafopy.printing.drop_stars(header, rows, signif_stars)
    return metafopy.printing.fmt_table(header, rows, row_names=None if estimate.shape[0] == 1 else list(names))


def test_line(stat: float, df: tuple[int, float | None], p: float, digits: int, name: str) -> str:
    if df[1] is not None:
        return f'F(df1 = {df[0]}, df2 = {int(df[1])}) = {metafopy.printing.fmt_num(stat, digits)}, p-val {metafopy.models.rma_uni.pfmt(p, digits)}'
    return f'{name}(df = {df[0]}) = {metafopy.printing.fmt_num(stat, digits)}, p-val {metafopy.models.rma_uni.pfmt(p, digits)}'


def format_result(
    res: metafopy.models.rma_ls_result.RMALSResult, digits: int = 4, showfit: bool = False, signif_stars: bool = True, signif_legend: bool = True
) -> str:
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    out: list[str] = ['']
    out.append(f'Location-Scale Model (k = {res.k}; tau^2 estimator: {res.method})')
    out.append('')
    if showfit:
        st = res.fit_stats(None)
        cells = [metafopy.printing.fmt_num(v, digits) for v in st.as_tuple()]
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        out.append(metafopy.printing.fmt_named_vector(['logLik', 'deviance', 'AIC', 'BIC', 'AICc'], cells))
        out.append('')

    if not math.isnan(res.QE):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        name, label = ('Q', 'Heterogeneity') if res.int_only else ('QE', 'Residual Heterogeneity')
        out.append(f'Test for {label}:')
        out.append(f'{name}(df = {res.k - res.p}) = {metafopy.printing.fmt_num(res.QE, digits)}, p-val {metafopy.models.rma_uni.pfmt(res.QEp, digits)}')
        out.append('')

    if res.p > 1 and not math.isnan(res.QM):
        coefs = metafopy.models.rma_uni.coef_span(res.btt)
        out.append(f'Test of Location Coefficients (coefficient{"s" if res.m > 1 else ""} {coefs}):')
        out.append(test_line(res.QM, res.QMdf, res.QMp, digits, 'QM'))
        out.append('')

    out.append('Model Results (Location):')
    out.append('')
    out.append(coef_table(res.beta, res.se, res.zval, res.pval, res.ci_lb, res.ci_ub, res.coef_names, res.ddf, None, digits, signif_stars))
    out.append('')

    if res.q > 1 and not math.isnan(res.QS):
        coefs = metafopy.models.rma_uni.coef_span(res.att)
        out.append(f'Test of Scale Coefficients (coefficient{"s" if res.m_alpha > 1 else ""} {coefs}):')
        out.append(test_line(res.QS, res.QSdf, res.QSp, digits, 'QM'))
        out.append('')

    out.append('Model Results (Scale):')
    out.append('')
    fixed = numpy.where(res.alpha_fix, res.alpha, math.nan)
    out.append(
        coef_table(
            res.alpha,
            res.se_alpha,
            res.zval_alpha,
            res.pval_alpha,
            res.ci_lb_alpha,
            res.ci_ub_alpha,
            res.alpha_names,
            res.ddf_alpha,
            fixed,
            digits,
            signif_stars,
        )
    )
    out += metafopy.printing.signif_footer(signif_legend)
    return '\n'.join(out)
