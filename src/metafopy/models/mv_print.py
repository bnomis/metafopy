"""Text output of an ``rma.mv`` fit, after metafor's ``print.rma.mv``.

https://github.com/wviechtb/metafor/blob/master/R/print.rma.mv.r
"""

import collections.abc
import math
import re

import numpy
import numpy.typing

import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_uni
import metafopy.models.robust_print
import metafopy.printing

__all__ = ['abbreviate', 'format_result']

type FloatArray = numpy.typing.NDArray[numpy.float64]

SPATIAL_OR_GENERAL = (*metafopy.models.mv_structs.SPATIAL, 'GDIAG', 'GEN')


def abbreviate_once(text: str, minlen: int) -> str:
    """R's ``abbreviate()`` for one string (``use.classes=TRUE``, ``method="left.kept"``); the C function ``stripchars``."""
    buff = list(text.strip())
    if len(buff) <= minlen:
        return ''.join(buff)
    vowels = set('aeiou')

    def space(i: int) -> bool:
        return buff[i].isspace()

    nspace = 0
    for i in range(len(buff) - 1, 0, -1):
        if space(i):
            nspace += 1
        if len(buff) - nspace <= minlen:
            return squeeze(buff, minlen)

    def first(i: int) -> bool:
        return space(i - 1)

    def last(i: int) -> bool:
        return (not space(i - 1)) and (i + 1 >= len(buff) or space(i + 1))

    def done() -> bool:
        return len(buff) - nspace <= minlen

    tests: list[collections.abc.Callable[[int], bool]] = [
        lambda i: buff[i] in vowels and last(i),
        lambda i: buff[i] in vowels and not first(i),
        lambda i: buff[i].islower() and last(i),
        lambda i: buff[i].islower() and not first(i),
    ]
    for test in tests:
        i = len(buff) - 1
        while i > 0:
            if i < len(buff) and test(i):
                del buff[i]
            if done():
                return squeeze(buff, minlen)
            i -= 1
    i = len(buff) - 1
    while i > 0:
        if i < len(buff) and not first(i) and not space(i):
            del buff[i]
        if done():
            return squeeze(buff, minlen)
        i -= 1
    return squeeze(buff, minlen)


def squeeze(buff: list[str], minlen: int) -> str:
    out = list(buff)
    if len(out) > minlen:
        out = [c for j, c in enumerate(out) if j == 0 or not c.isspace()]
    return ''.join(out)


def abbreviate(names: list[str], minlength: int = 4) -> list[str]:
    """R's ``abbreviate(names, minlength)``: lengthens the abbreviation of duplicates until they are unique."""
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    result = [''] * len(names)
    dup = [True] * len(names)
    these = list(names)
    while True:
        ans = [abbreviate_once(t, minlength) for t in these]
        pos = [i for i, d in enumerate(dup) if d]
        for i, a in zip(pos, ans, strict=True):
            result[i] = a
        seen: set[str] = set()
        newdup = [False] * len(names)
        for i, a in enumerate(result):
            if a in seen:
                newdup[i] = True
            seen.add(a)
        if not any(newdup):
            return result
        minlength += 1
        flagged = {result[i] for i, d in enumerate(newdup) if d}
        dup = [newdup[i] or result[i] in flagged for i in range(len(names))]
        these = [names[i] for i, d in enumerate(dup) if d]


def lower_pairs(n: int) -> list[tuple[int, int]]:
    """Row and column of the strictly lower triangle in R's column-major order."""
    return [(i, j) for j in range(n) for i in range(j + 1, n)]


def yes_no(flags: numpy.typing.NDArray[numpy.bool_] | None, n: int) -> list[str]:
    if flags is None:
        return ['no'] * n
    return ['yes' if f else 'no' for f in flags]


def inner_text(group: metafopy.models.mv_structs.Group) -> str:
    inner = group.formula.inner or ''
    inner = re.sub(r'\s*\+\s*', ' + ', inner.strip())
    text = '~' + inner
    return text if len(text) <= 15 else text[:15] + '[...]'


def block(res: metafopy.models.mv_result.RMAMVResult, group: metafopy.models.mv_structs.Group, is_g: bool, digits: int) -> list[str]:
    """The printed variance components of one ``~ inner | outer`` term."""
    out: list[str] = []
    struct = group.struct
    names = res.g_names if is_g else res.h_names
    v_name, r_name, s_name = ('tau^2', 'rho', 'tau') if is_g else ('gamma^2', 'phi', 'gamma')
    var = res.tau2 if is_g else res.gamma2
    cor = res.rho if is_g else res.phi
    cov = res.G if is_g else res.H
    fix_v = res.vc_fix['tau2' if is_g else 'gamma2']
    fix_r = res.vc_fix['rho' if is_g else 'phi']
    if var is None or cor is None:
        return out

    def fm(x: float) -> str:
        return metafopy.printing.fmt_num(float(x), digits)

    estim = [fm(x) for x in var]
    sqrt = [fm(math.sqrt(x)) for x in var]
    cor_s = [fm(x) for x in cor]
    levels_f = group.levels_f[0]
    levels_k = group.levels_k if group.levels_k is not None else numpy.zeros(len(levels_f))
    k_lvl = [metafopy.printing.fmt_num(float(x), 0) for x in levels_k]
    inner_names = names[:-1]
    outer = names[-1]
    inner = inner_text(group) if struct in SPATIAL_OR_GENERAL else names[0]
    width = max(len(inner), len(outer))
    kind = 'inner term:  ' if struct in SPATIAL_OR_GENERAL else 'inner factor:'
    out.append(f'outer factor: {outer.ljust(width)} (nlvls = {group.nlevels[1]})')
    out.append(f'{kind} {inner.ljust(width)} (nlvls = {group.nlevels_f[0]})')
    out.append('')
    fixed_v = yes_no(fix_v, len(var))
    fixed_r = yes_no(fix_r, len(cor))
    if struct in ('CS', 'AR', 'CAR', 'ID', *metafopy.models.mv_structs.SPATIAL):
        rows = [[estim[0], sqrt[0], fixed_v[0]], [cor_s[0], '', fixed_r[0]]]
        names_r = [f'{v_name}    ' if is_g else f'{v_name}  ', r_name]
        if struct == 'ID':
            rows, names_r = rows[:1], names_r[:1]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        out.append(metafopy.printing.fmt_table(['estim', 'sqrt', 'fixed'], rows, row_names=names_r))
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    elif struct in ('HCS', 'HAR', 'DIAG'):
        rows = [[estim[i], sqrt[i], k_lvl[i], fixed_v[i], levels_f[i]] for i in range(len(var))]
        rows.append([cor_s[0], '', '', fixed_r[0], ''])
        if len(var) == 1:
            names_r = [f'{v_name}   ' if is_g else f'{v_name} ', r_name]
        else:
            names_r = [f'{v_name}.{i + 1}  ' for i in range(len(var))] + [r_name]
        if struct == 'DIAG':
            rows, names_r = rows[: len(var)], names_r[: len(var)]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        out.append(metafopy.printing.fmt_table(['estim', 'sqrt', 'k.lvl', 'fixed', 'level'], rows, row_names=names_r))
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    elif struct == 'UN':
        rows = [[estim[i], sqrt[i], k_lvl[i], fixed_v[i], levels_f[i]] for i in range(len(var))]
        names_r = [v_name] if len(levels_k) == 1 else [f'{v_name}.{i + 1}  ' for i in range(len(levels_k))]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        out.append(metafopy.printing.fmt_table(['estim', 'sqrt', 'k.lvl', 'fixed', 'level'], rows, row_names=names_r))
        out.append('')
        n = 2 if len(cor) == 1 else group.nlevels_f[0]
        pairs = lower_pairs(n)
        comb = group.levels_comb_k if group.levels_comb_k is not None else numpy.zeros(len(pairs))
        lower = {ij: cor_s[pos] for pos, ij in enumerate(pairs)}
        info = {ij: fixed_r[pos] for pos, ij in enumerate(pairs)}
        counts = {ij: metafopy.printing.fmt_num(float(comb[pos]), 0) for pos, ij in enumerate(pairs)}
        cells: list[list[str]] = []
        for i in range(n):
            row = [('1' if i == j else lower[i, j] if j < i else '') for j in range(n)]
            row.append('')
            row += ['-' if i == j else info[i, j] if j < i else counts[j, i] for j in range(n)]
            cells.append(row)
        ab = abbreviate(list(levels_f))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        header = [f'{r_name}.{a}' for a in ab] + [''] + ab
        out.append(metafopy.printing.fmt_table(header, cells, row_names=list(levels_f)))
    elif struct == 'GEN':
        if cov is None:
            return out
        cor_mat = metafopy.models.mv_structs.cov2cor(cov)
        p = len(inner_names)
        header = ['estim', 'sqrt', 'fixed', f'{r_name}:', *abbreviate(list(inner_names))]
        low_idx = lower_pairs(p)
        info_flags = {ij: fixed_r[pos] for pos, ij in enumerate(low_idx)}
        rows = []
        for i in range(p):
            row = [estim[i], sqrt[i], fixed_v[i], '']
            for j in range(p):
                if i == j:
                    row.append('-')
                elif j < i:
                    row.append(info_flags.get((i, j), 'no'))
                else:
                    row.append(fm(cor_mat[i, j]))
            rows.append(row)
        out.append(metafopy.printing.fmt_table(header, rows, row_names=list(inner_names)))
    elif struct == 'GDIAG':
        rows = [[estim[i], sqrt[i], fixed_v[i]] for i in range(len(var))]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        out.append(metafopy.printing.fmt_table(['estim', 'sqrt', 'fixed'], rows, row_names=list(inner_names)))
    out.append('')
    del s_name
    return out


def format_result(
    res: metafopy.models.mv_result.RMAMVResult, digits: int = 4, showfit: bool = False, signif_stars: bool = True, signif_legend: bool = True
) -> str:
    fm = metafopy.printing.fmt_num
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    out: list[str] = ['']
    out.append(f'Multivariate Meta-Analysis Model (k = {res.k}; method: {res.method})')
    if showfit:
        st = res.fit_stats_reml if res.method == 'REML' else res.fit_stats_ml
        out.append('')
        out.append(metafopy.printing.fmt_named_vector(['logLik', 'Deviance', 'AIC', 'BIC', 'AICc'], [fm(v, digits) for v in st.as_tuple()]))
        out.append('')
    else:
        out.append('')
    if not (res.with_s or res.with_g or res.with_h):
        out.append('Variance Components: none')
        out.append('')
    else:
        out.append('Variance Components:')
        out.append('')
        if res.with_s and res.sigma2 is not None:
            n = len(res.sigma2)
            header = ['estim', 'sqrt', 'nlvls', 'fixed', 'factor']
            if res.with_r:
                header.append('R')
            fixed = yes_no(res.vc_fix['sigma2'], n)
            rows = []
            for j in range(n):
                row = [fm(res.sigma2[j], digits), fm(math.sqrt(res.sigma2[j]), digits), str(res.s_nlevels[j]), fixed[j], res.s_names[j]]
                if res.with_r:
                    row.append('yes' if res.Rfix[j] else 'no')
                rows.append(row)
            names = ['sigma^2  '] if n == 1 else [f'sigma^2.{j + 1}' for j in range(n)]
            out.append(metafopy.printing.fmt_table(header, rows, row_names=names))
            out.append('')
        if res.with_g and res.group_g is not None:
            out.extend(block(res, res.group_g, True, digits))
        if res.with_h and res.group_h is not None:
            out.extend(block(res, res.group_h, False, digits))

    if not math.isnan(res.QE):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        name, label = ('Q', 'Heterogeneity') if res.int_only else ('QE', 'Residual Heterogeneity')
        out.append(f'Test for {label}:')
        out.append(f'{name}(df = {res.QEdf}) = {fm(res.QE, digits)}, p-val {metafopy.models.rma_uni.pfmt(res.QEp, digits)}')
        out.append('')

    robust = res.robust
    if robust is not None:
        out += metafopy.models.robust_print.counts_lines(robust, res.k)

    if res.p > 1 and not math.isnan(res.QM):
        coefs = metafopy.models.rma_uni.coef_span(res.btt)
        mark = metafopy.models.robust_print.FOOT_MARK if robust is not None else ''
        out.append(f'Test of Moderators (coefficient{"s" if res.m > 1 else ""} {coefs}):{mark}')
        qm_p = metafopy.models.rma_uni.pfmt(res.QMp, digits)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if res.test in ('knha', 'adhoc', 't'):
            df2 = metafopy.printing.fmt_df_scalar(res.QMdf[1])
            out.append(f'F(df1 = {res.QMdf[0]}, df2 = {df2}) = {fm(res.QM, digits)}, p-val {qm_p}')
        else:
            out.append(f'QM(df = {res.QMdf[0]}) = {fm(res.QM, digits)}, p-val {qm_p}')
        out.append('')

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    use_t = res.test in ('knha', 'adhoc', 't')
    header = ['estimate', 'se', 'tval' if use_t else 'zval']
    if use_t:
        header.append('df')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    header += ['pval', 'ci.lb', 'ci.ub', '']
    df_cells = (metafopy.models.robust_print.df_cells(res.ddf) if robust is not None else metafopy.printing.fmt_df(res.ddf)) if use_t else []
    rows_: list[list[str]] = []
    for j in range(res.p):
        row = [fm(res.beta[j], digits), fm(res.se[j], digits), fm(res.zval[j], digits)]
        if use_t:
            row.append(df_cells[j])
        row += [
            metafopy.printing.fmt_pval(res.pval[j], digits),
            fm(res.ci_lb[j], digits),
            fm(res.ci_ub[j], digits),
            metafopy.printing.signif_stars(res.pval[j]),
        ]
        rows_.append(row)
    if robust is not None:
        header, rows_ = metafopy.models.robust_print.decorate(header, rows_)
    header, rows_ = metafopy.printing.drop_stars(header, rows_, signif_stars)
    out.append('Model Results:')
    out.append('')
    out.append(
        metafopy.printing.fmt_table(
            header, rows_, row_names=None if res.int_only else list(res.coef_names), width=metafopy.printing.console_width(1 if robust is not None else 0)
        )
    )
    out += metafopy.printing.signif_footer(signif_legend, metafopy.models.robust_print.legend_lines(robust, res.int_only) if robust is not None else ())
    return '\n'.join(out)
