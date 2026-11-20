"""``published`` cases: expected values printed in the defining publication or a textbook worked example."""

import collections.abc
import typing

import numpy
import pytest

import correctness.cases
import metafopy


@pytest.mark.parametrize('case', correctness.cases.case_params('published'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_published(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def scalar(value: typing.Any) -> float:
    return float(numpy.asarray(value, dtype=numpy.float64).reshape(-1)[0])


def check_values(case: correctness.cases.Case, values: collections.abc.Mapping[str, float]) -> None:
    unknown = set(case.printed) - set(values)
    if unknown:
        raise LookupError(f'{case.id}: this check does not know the printed name(s) {sorted(unknown)}; it knows {sorted(values)}')
    for name in case.printed:
        correctness.cases.check_printed(case, name, values[name])


def back_transformed(case: correctness.cases.Case, result: typing.Any) -> None:
    """The pooled ratio and its confidence bounds, exp(beta), exp(ci_lb) and exp(ci_ub), as a source prints them (Mantel and Haenszel, 1959, Table 1)."""
    check_values(
        case,
        {
            'exp_beta': float(numpy.exp(scalar(result.beta))),
            'exp_ci_lb': float(numpy.exp(scalar(result.ci_lb))),
            'exp_ci_ub': float(numpy.exp(scalar(result.ci_ub))),
        },
    )


def prediction_interval(case: correctness.cases.Case, result: typing.Any) -> None:
    """The prediction interval of the pooled effect, ``predict(result, predtype='riley')`` (Borenstein et al., 2009, chapter 17, equations 17.7 and 17.8).

    The book takes t on k - 2 degrees of freedom, which is the ``'riley'`` interval of an intercept-only fit with the t test.
    """
    predicted = metafopy.predict(result, predtype='riley')
    values = {'pred': scalar(predicted.pred), 'pi_lb': scalar(predicted.pi_lb), 'pi_ub': scalar(predicted.pi_ub)}
    values.update({f'exp_{name}': float(numpy.exp(value)) for name, value in list(values.items())})
    check_values(case, values)


def confint_ht(case: correctness.cases.Case, result: typing.Any) -> None:
    """The test-based intervals of tau2, I2, H2 and H, ``confint(result, type='ht')`` (Higgins and Thompson, 2002; Borenstein et al., 2009, eqs 16.14-16.19)."""
    table = metafopy.confint(result, type='ht').random
    if table is None:
        raise LookupError(f'{case.id}: confint gives no random-effects table')
    values: dict[str, float] = {}
    for row, name in (('tau^2', 'tau2'), ('I^2(%)', 'I2'), ('H^2', 'H2')):
        values[f'{name}_lb'] = float(table.loc[row, 'ci.lb'])
        values[f'{name}_ub'] = float(table.loc[row, 'ci.ub'])
    values['H_lb'] = float(numpy.sqrt(values['H2_lb']))
    values['H_ub'] = float(numpy.sqrt(values['H2_ub']))
    check_values(case, values)


def glmm_printed(case: correctness.cases.Case, result: typing.Any) -> None:
    """The coefficients of an ``rma_glmm`` fit on the scales a source prints.

    ``beta[j]``, ``se[j]``, ``ci_lb[j]`` and ``ci_ub[j]`` are the fit's; ``exp_``, ``neg_`` and ``expit_`` prefix the odds-ratio scale, the opposite
    sign (a source that models control against treatment) and a proportion; ``tau`` is the between-study standard deviation, ``tau2`` the variance.
    """
    values: dict[str, float] = {'tau2': scalar(result.tau2), 'tau': float(numpy.sqrt(scalar(result.tau2)))}
    for index in range(numpy.asarray(result.beta).size):
        for name in ('beta', 'se', 'ci_lb', 'ci_ub'):
            raw = float(numpy.asarray(getattr(result, name), dtype=numpy.float64).reshape(-1)[index])
            values[f'{name}[{index}]'] = raw
            values[f'neg_{name}[{index}]'] = -raw
            values[f'exp_{name}[{index}]'] = float(numpy.exp(raw))
            values[f'expit_{name}[{index}]'] = float(1.0 / (1.0 + numpy.exp(-raw)))
    check_values(case, values)


def vl2022_wald(case: correctness.cases.Case, result: typing.Any) -> None:
    """The coefficients of a location-scale fit with their Wald-type intervals and p-values, and its omnibus tests (Viechtbauer and Lopez-Lopez, 2022).

    ``beta[j]``, ``ci_lb[j]``, ``ci_ub[j]``, ``pval[j]`` are the location part's, ``alpha[j]``, ``ci_lb_alpha[j]``, ``ci_ub_alpha[j]``, ``pval_alpha[j]`` the
    scale part's. ``QM``, ``QMp`` and ``QS``, ``QSp`` are the omnibus tests of all non-intercept coefficients of each part; ``QM_subject``,
    ``QMp_subject`` and ``QS_subject``, ``QSp_subject`` those of the last two coefficients (the subject type), by ``anova(btt=)`` and ``anova(att=)``.
    """
    values: dict[str, float] = {}
    for part in ('beta', 'ci_lb', 'ci_ub', 'pval', 'alpha', 'ci_lb_alpha', 'ci_ub_alpha', 'pval_alpha'):
        for index, raw in enumerate(numpy.asarray(getattr(result, part), dtype=numpy.float64).reshape(-1)):
            values[f'{part}[{index}]'] = float(raw)
    values.update({'QM': float(result.QM), 'QMp': float(result.QMp), 'QS': float(result.QS), 'QSp': float(result.QSp)})
    last_beta, last_alpha = numpy.asarray(result.beta).size, numpy.asarray(result.alpha).size
    location = metafopy.anova(result, btt=[last_beta - 2, last_beta - 1])
    scale = metafopy.anova(result, att=[last_alpha - 2, last_alpha - 1])
    values.update(
        {
            'QM_subject': float(location.stat),
            'QMp_subject': float(location.pval_omnibus),
            'QS_subject': float(scale.stat),
            'QSp_subject': float(scale.pval_omnibus),
        }
    )
    check_values(case, values)


def vl2022_profile(case: correctness.cases.Case, result: typing.Any) -> None:
    """The profile-likelihood interval of each scale coefficient, ``confint(result, alpha=j)``, as ``plci_lb[j]`` and ``plci_ub[j]`` (0-based ``j``)."""
    values: dict[str, float] = {}
    for index in range(numpy.asarray(result.alpha).size):
        table = metafopy.confint(result, alpha=index).random
        if table is None:
            raise LookupError(f'{case.id}: confint gives no random-effects table')
        values[f'plci_lb[{index}]'] = float(table['ci.lb'].iloc[0])
        values[f'plci_ub[{index}]'] = float(table['ci.ub'].iloc[0])
    check_values(case, values)


def vl2022_fitstats(case: correctness.cases.Case, result: typing.Any) -> None:
    """Log-likelihood, AIC, BIC and AICc at the fit's own method (Viechtbauer and Lopez-Lopez, 2022, Table 3)."""
    stats = result.fit_stats_reml if case.args['method'] == 'REML' else result.fit_stats_ml
    check_values(case, {'logLik': float(stats.ll), 'AIC': float(stats.AIC), 'BIC': float(stats.BIC), 'AICc': float(stats.AICc)})


def vl2022_predict_location(case: correctness.cases.Case, result: typing.Any) -> None:
    """The predicted effect and its 95% prediction interval for a math study of 50, 100 and 150 participants: ``pred_<n>``, ``pi_lb_<n>``, ``pi_ub_<n>``."""
    values: dict[str, float] = {}
    for participants in (50, 100, 150):
        size = participants / 100
        predicted = metafopy.predict(result, newmods=[size, 0, 0], newscale=[size, 0, 0])
        values[f'pred_{participants}'] = scalar(predicted.pred)
        values[f'pi_lb_{participants}'] = scalar(predicted.pi_lb)
        values[f'pi_ub_{participants}'] = scalar(predicted.pi_ub)
    check_values(case, values)


def vl2022_predict_tau2(case: correctness.cases.Case, result: typing.Any) -> None:
    """The predicted ``tau2`` with its 95% interval at 100 participants for math, social science and science studies: ``math``, ``math_lb``, ``math_ub``, ..."""
    values: dict[str, float] = {}
    for subject, newscale in (('math', [1, 0, 0]), ('soc', [1, 0, 1]), ('sci', [1, 1, 0])):
        predicted = metafopy.predict(result, newscale=newscale, transf=numpy.exp)
        values[subject] = scalar(predicted.pred)
        values[f'{subject}_lb'] = scalar(predicted.ci_lb)
        values[f'{subject}_ub'] = scalar(predicted.ci_ub)
    check_values(case, values)


def vl2022_lrt(case: correctness.cases.Case, result: typing.Any) -> None:
    """The likelihood ratio test (equation 15) of the full ML model against sample size in the location and subject type in the scale."""
    reduced = metafopy.rma_uni(
        yi='yi',
        vi='vi',
        data=correctness.cases.case_frame(case),
        mods='~ ni100',
        scale='~ sci + soc',
        method='ML',
        test='knha',
    )
    test = metafopy.anova(result, reduced)
    check_values(case, {'LRT': float(test.LRT), 'pval': float(test.pval), 'df': float(test.df_full - test.df_reduced)})


def influence_printed(case: correctness.cases.Case, result: typing.Any) -> None:
    """``influence(fit)`` values a source prints (Viechtbauer and Cheung, 2010, Table I).

    ``hat[i]``, ``rstudent[i]``, ``cook.d[i]`` and ``dfbetas[i,j]`` (0-based study ``i`` and coefficient ``j``) are the fit's; ``R[i]`` is
    ``100 (tau2 - tau2.del[i]) / tau2``, the percentage change in ``tau2`` when study ``i`` is left out; ``tau2`` is the fit's.
    """
    influence = metafopy.influence(result)
    values: dict[str, float] = {'tau2': scalar(result.tau2)}
    for i in range(len(influence.hat)):
        values[f'hat[{i}]'] = float(influence.hat[i])
        values[f'rstudent[{i}]'] = float(influence.rstudent[i])
        values[f'cook.d[{i}]'] = float(influence.cook_d[i])
        values[f'R[{i}]'] = 100 * (scalar(result.tau2) - float(influence.tau2_del[i])) / scalar(result.tau2)
        for j in range(influence.dfbs_array.shape[1]):
            values[f'dfbetas[{i},{j}]'] = float(influence.dfbs_array[i, j])
    check_values(case, values)


def blup_printed(case: correctness.cases.Case, result: typing.Any) -> None:
    """``blup(fit)`` values a source prints (Raudenbush, 2009, Table 16.2): ``pred[i]`` (0-based study), ``min(pred)`` and ``max(pred)``."""
    predicted = metafopy.blup(result).pred
    values = {f'pred[{i}]': float(value) for i, value in enumerate(predicted)}
    values.update({'min(pred)': float(predicted.min()), 'max(pred)': float(predicted.max())})
    check_values(case, values)
