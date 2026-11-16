#!/usr/bin/env python3
"""Measure what each numerical failure group of the cross-platform numerics spec compares, in this environment: ``bin/platform-probe.sh``.

Each group of ``agents/specs/cross-platform-numerics/spec.md`` (Failure groups) has one measuring function that calls the helpers its tests call, so the
figures are those the tests see. The output is JSON with the identity of the environment and one ``GroupMeasurement`` per group; outputs from several
environments are compared by hand and the figures copied into ``measurements.md``.
"""

import argparse
import collections.abc
import contextlib
import dataclasses
import json
import math
import pathlib
import platform
import re
import sys
import typing
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'tests'))

import correctness.cases  # noqa: E402
import correctness.formulas  # noqa: E402
import correctness.glmm_formulas  # noqa: E402
import correctness.test_derived  # noqa: E402
import correctness.test_properties  # noqa: E402
import numpy  # noqa: E402
import scipy  # noqa: E402
import scipy.optimize  # noqa: E402
import test_audit4_numerics  # noqa: E402
import test_nchg  # noqa: E402
import test_rma_ls  # noqa: E402
import test_selmodel  # noqa: E402
import test_stepcon  # noqa: E402

import metafopy  # noqa: E402
import metafopy.core.nchg  # noqa: E402

FloatArray = numpy.typing.NDArray[numpy.float64]
Values = dict[str, list[float]]

LS_BCG_CASES = ('ls.fit.ML.log.bcg', 'ls.fit.REML.log.bcg')
LS_IDENTITY_CASES = ('ls.fit.ML.identity.bcg', 'ls.fit.REML.identity.bcg')
SELMODEL_DECREASING_CASE = 'selmodel.decreasing.bangertdrowns2004'
GLMM_CASES = ('glmm.cor.um.rs.ml.bcg', 'glmm.fit.or.um.rs.ml.jackson2018.pneumonia')
MV_CASE = 'property.mv.reml-score.kon'
GLMM_STEPS = (2e-5, 5e-5, 1e-4, 2e-4, 4e-4)
RAY_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclasses.dataclass(frozen=True)
class GroupMeasurement:
    """What one failure group's tests compare, measured in this environment: named arrays, optimiser messages, and the error if the group raised."""

    group: str
    values: dict[str, list[float]]
    messages: list[str]
    error: str | None


@dataclasses.dataclass
class Record:
    """What a measuring function collects: the values and messages of one group."""

    values: Values = dataclasses.field(default_factory=dict)
    messages: list[str] = dataclasses.field(default_factory=list)

    def put(self, name: str, value: typing.Any) -> None:
        self.values[name] = [float(v) for v in numpy.ravel(numpy.asarray(value, dtype=numpy.float64))]

    def say(self, text: str) -> None:
        self.messages.append(text)


def environment() -> dict[str, str]:
    """``sys.platform``, ``platform.machine()``, the Python version and build, numpy and scipy versions, the BLAS and LAPACK libraries from
    ``numpy.show_config(mode='dicts')``, ``platform.libc_ver()`` on Linux and the OS release: the identity of the numbers that follow."""
    config = numpy.show_config(mode='dicts')
    blas = config['Build Dependencies']['blas']
    lapack = config['Build Dependencies']['lapack']
    out = {
        'sys.platform': sys.platform,
        'machine': platform.machine(),
        'python': platform.python_version(),
        'python_build': ' '.join(platform.python_build()) + f' {platform.python_compiler()}',
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'blas': f'{blas.get("name")} {blas.get("version")}',
        'lapack': f'{lapack.get("name")} {lapack.get("version")}',
        'os': platform.platform(),
    }
    if sys.platform.startswith('linux'):
        out['libc'] = ' '.join(platform.libc_ver())
        with contextlib.suppress(OSError):
            out['os_release'] = platform.freedesktop_os_release().get('PRETTY_NAME', '')
    return out


def case_by_id(case_id: str) -> correctness.cases.Case:
    for case in correctness.cases.load_cases():
        if case.id == case_id:
            return case
    raise KeyError(case_id)


def score_norm(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> float:
    """The largest absolute analytic score of ``ls_loglik`` at ``alpha``, or NaN where the model is not defined there."""
    try:
        return float(numpy.max(numpy.abs(correctness.formulas.ls_score(yi, vi, X, Z, alpha, link, method))))
    except ValueError:
        return math.nan


def ls_runs(record: Record, label: str, yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, link: str, method: str) -> None:
    """Every start of ``ls_reference_fit``, run as it runs them: message, ``-fun``, analytic score and ``alpha``; then the reference's own result."""

    def negative(alpha: FloatArray) -> float:
        if link == 'identity' and float(numpy.min(Z @ alpha)) < -correctness.formulas.IDENTITY_SLACK:
            return math.inf
        value = correctness.formulas.ls_loglik(yi, vi, X, Z, alpha, link, method)
        return math.inf if value == -math.inf else -value

    def gradient(alpha: FloatArray) -> FloatArray:
        if correctness.formulas.ls_components(yi, vi, X, Z, alpha, link) is None:
            return numpy.full(len(alpha), numpy.nan)
        return -correctness.formulas.ls_score(yi, vi, X, Z, alpha, link, method)

    starts = correctness.formulas.ls_starts(yi, vi, X, Z, link)
    if link == 'identity':
        starts.extend(correctness.formulas.ls_identity_starts(yi, vi, X, Z, method))
    for number, start in enumerate(starts):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if link == 'log':
                res = scipy.optimize.minimize(negative, start, jac=gradient, method='BFGS', options={'maxiter': 2000, 'gtol': 1e-9})
            else:
                feasible = scipy.optimize.LinearConstraint(Z, 0.0, numpy.inf)
                res = scipy.optimize.minimize(negative, start, jac=gradient, method='SLSQP', constraints=[feasible], options={'maxiter': 2000, 'ftol': 1e-14})
        name = f'{label}.start{number}'
        record.put(f'{name}.alpha', res.x)
        record.put(f'{name}.negloglik', res.fun)
        record.put(f'{name}.score', score_norm(yi, vi, X, Z, res.x, link, method))
        record.put(f'{name}.success', float(bool(res.success)))
        record.say(f'{name}: {res.message}')
    try:
        reference = correctness.formulas.ls_reference_fit(yi, vi, X, Z, link, method)
    except RuntimeError as error:
        # auditme: accept[except-interpolation] the probe records the optimiser's message as a figure; it is the measurement
        record.say(f'{label}.reference: RuntimeError: {error}')
        return
    record.put(f'{label}.reference.alpha', reference['alpha'])
    record.put(f'{label}.reference.ll', reference['ll'])
    record.put(f'{label}.reference.score', score_norm(yi, vi, X, Z, reference['alpha'], link, method))


def ls_case_group(case_ids: collections.abc.Sequence[str]) -> Record:
    record = Record()
    for case_id in case_ids:
        case = case_by_id(case_id)
        yi, vi, X, Z = correctness.test_derived.ls_inputs(case)
        link, method = case.args.get('link', 'log'), case.args['method']
        result = correctness.cases.run_case(case)
        record.put(f'{case_id}.metafopy.alpha', result.alpha)
        record.put(f'{case_id}.metafopy.ll', float(result.logLik(REML=method == 'REML')))
        record.put(f'{case_id}.metafopy.score', score_norm(yi, vi, X, Z, numpy.asarray(result.alpha, dtype=numpy.float64), link, method))
        ls_runs(record, case_id, yi, vi, X, Z, link, method)
    return record


def measure_ls_bcg_reference() -> Record:
    return ls_case_group(LS_BCG_CASES)


def measure_ls_identity_alpha() -> Record:
    return ls_case_group(LS_IDENTITY_CASES)


def measure_ls_identity_maximum() -> Record:
    record = Record()
    res = test_rma_ls.FITS['bcg_year_identity']()
    yi, vi, X, Z = (numpy.asarray(v, dtype=numpy.float64) for v in (res.yi, res.vi, res.X, res.Z))
    record.put('metafopy.alpha', res.alpha)
    record.put('metafopy.ll', res.fit_stats_ml.ll if res.method == 'ML' else res.fit_stats_reml.ll)
    record.put('metafopy.score', score_norm(yi, vi, X, Z, numpy.asarray(res.alpha, dtype=numpy.float64), 'identity', res.method))
    ls_runs(record, 'bcg_year_identity', yi, vi, X, Z, 'identity', res.method)
    return record


@contextlib.contextmanager
def captured_minimize() -> collections.abc.Iterator[list[tuple[collections.abc.Callable[..., float], scipy.optimize.OptimizeResult]]]:
    """Every ``scipy.optimize.minimize`` call made inside the block: its objective and its result, whether or not it converged."""
    calls: list[tuple[collections.abc.Callable[..., float], scipy.optimize.OptimizeResult]] = []
    original = scipy.optimize.minimize

    def wrapper(fun: collections.abc.Callable[..., float], x0: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> scipy.optimize.OptimizeResult:
        res: scipy.optimize.OptimizeResult = original(fun, x0, *args, **kwargs)
        calls.append((fun, res))
        return res

    scipy.optimize.minimize = wrapper  # type: ignore[assignment]
    try:
        yield calls
    finally:
        scipy.optimize.minimize = original


def measure_selmodel_decreasing() -> Record:
    record = Record()
    case = case_by_id(SELMODEL_DECREASING_CASE)
    result = correctness.cases.run_case(case)
    free = ~numpy.asarray(result.delta_fix, dtype=bool)
    point = numpy.concatenate([numpy.ravel(result.beta), [float(result.tau2)], numpy.asarray(result.delta, dtype=numpy.float64)[free]])
    record.put('metafopy.point', point)
    record.put('metafopy.ll', float(result.ll))
    with captured_minimize() as calls:
        try:
            beta, tau2, delta, loglik, plain = correctness.test_derived.selmodel_maximum(case, result)
            record.put('reference.beta', beta)
            record.put('reference.tau2', tau2)
            record.put('reference.delta', delta)
            record.put('reference.ll', loglik)
            record.put('reference.ll_without_selection', plain)
        except AssertionError as error:
            # auditme: accept[except-interpolation] the probe records the optimiser's message as a figure; it is the measurement
            record.say(f'selmodel_maximum: AssertionError: {error}')
    objective, fit = calls[0]
    record.put('reference.end_point', fit.x)
    record.put('reference.negloglik', fit.fun)
    record.say(f'reference.slsqp: {fit.message}')
    record.put('ll_at_metafopy_point', -objective(point))
    end = numpy.asarray(fit.x, dtype=numpy.float64)
    record.put('ll_along_ray', [-objective(point + fraction * (end - point)) for fraction in RAY_FRACTIONS])
    return record


def glmm_theta(case: correctness.cases.Case) -> tuple[correctness.glmm_formulas.Structure, correctness.glmm_formulas.Scalar, FloatArray]:
    frame = correctness.cases.case_frame(case)
    problem, rule, points = correctness.cases.formula_value(case, 'glmm_formulas.rma_glmm_problem', frame)
    method = case.args.get('method', 'ML')
    st, loglik = correctness.glmm_formulas.loglik_function(problem, method, rule, points)
    theta0 = numpy.concatenate([st.start, st.variance_start])
    return st, loglik, correctness.glmm_formulas.maximise(loglik, theta0, st)


def measure_glmm_um_rs_score() -> Record:
    record = Record()
    for case_id in GLMM_CASES:
        _, loglik, theta = glmm_theta(case_by_id(case_id))
        record.put(f'{case_id}.theta', theta)
        record.put(f'{case_id}.loglik', loglik(theta))
        record.put(f'{case_id}.score_richardson', numpy.max(numpy.abs(correctness.glmm_formulas.gradient_at(loglik, theta))))
        for step in GLMM_STEPS:
            record.put(f'{case_id}.score_step_{step:g}', numpy.max(numpy.abs(correctness.glmm_formulas.fd_gradient(loglik, theta, step))))
    return record


def measure_cm_el_tight_se() -> Record:
    record = Record()
    for frame, expected, name in zip(test_nchg.FRAMES, test_nchg.ML_SES, test_nchg.FRAME_IDS, strict=True):
        fit = metafopy.rma_glmm(**{**test_nchg.ARGS, 'method': 'ML'}, data=frame)
        tables, X = test_nchg.tables_of(frame), numpy.ones((len(frame), 1))
        start = numpy.append(fit.beta, math.log(fit.tau2))

        def objective(b: FloatArray, tables: metafopy.core.nchg.Tables = tables, X: FloatArray = X) -> float:
            return metafopy.core.nchg.neg_loglik(b, tables, X, True, test_nchg.TIGHT_RANDOM)

        result = scipy.optimize.minimize(objective, start, method='Nelder-Mead', options={'xatol': 1e-11, 'fatol': 1e-14, 'maxiter': 5000, 'maxfev': 10000})
        score = numpy.array([(objective(result.x + e) - objective(result.x - e)) / 2e-5 for e in numpy.eye(len(start)) * 1e-5])
        record.put(f'{name}.expected_se', expected)
        record.put(f'{name}.se_at_default_estimate', test_nchg.ml_se(frame, start))
        record.put(f'{name}.se_at_tight_optimum', test_nchg.ml_se(frame, result.x))
        record.put(f'{name}.end_point', result.x)
        record.put(f'{name}.default_estimate', start)
        record.put(f'{name}.negloglik', result.fun)
        record.put(f'{name}.score', score)
        record.say(f'{name}: {result.message} (nit {result.nit}, nfev {result.nfev})')
    return record


def measure_mv_reml_score() -> Record:
    record = Record()
    case = case_by_id(MV_CASE)
    fit = correctness.test_properties.refit_with(case, control=correctness.test_properties.MV_CONTROL)
    s = correctness.test_derived.mv_setup(case, fit)
    inverse = numpy.linalg.inv(s.V + s.extra())
    p = inverse - inverse @ s.X @ numpy.linalg.solve(s.X.T @ inverse @ s.X, s.X.T @ inverse)
    record.put('sigma2', fit.sigma2)
    scores, scales = [], []
    for term in s.terms:
        z = correctness.formulas.term_matrix(term, numpy.ones(1), numpy.zeros(0))
        scales.append(numpy.trace(p @ z))
        scores.append(s.yi @ p @ z @ p @ s.yi - numpy.trace(p @ z))
    record.put('score', scores)
    record.put('scale_trace_PZ', scales)
    record.put('relative_score', [a / b for a, b in zip(scores, scales, strict=True)])
    record.put('ll', fit.fit_stats_reml.ll)
    for name in ('converged', 'message', 'optimizer_message'):
        value = getattr(fit, name, None)
        if value is not None:
            record.say(f'fit.{name}: {value}')
    return record


def measure_stepcon_rau_mods() -> Record:
    record = Record()
    case = 'rau_stepcon_mods'
    res, ref = test_stepcon.fitted(case), test_stepcon.REF['fits'][case]
    for name in (*test_selmodel.WALD, 'se_tau2', 'QM', 'QMp', 'LRT', 'LRTp'):
        record.put(f'{name}', getattr(res, name))
        record.put(f'{name}.r', ref[name])
    record.put('delta', res.delta)
    record.put('delta.r', ref['delta'])
    record.put('beta', res.beta)
    record.put('tau2', res.tau2)
    record.put('ll', res.ll)
    record.put('ll.r', ref['ll'])
    record.put('vb', res.vb)
    record.put('vd', res.vd)
    return record


def draws(record: Record, label: str, res: typing.Any) -> None:
    for part in ('beta', 'alpha'):
        sims = numpy.asarray(getattr(res, part).sim, dtype=numpy.float64)
        record.put(f'{label}.{part}.nan_draws', numpy.flatnonzero(numpy.isnan(sims[:, 0])))
        record.put(f'{label}.{part}.first_draws', sims[:4, 0])


def measure_ls_vif_sim() -> Record:
    record = Record()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bangert = metafopy.rma(yi='yi', vi='vi', mods='~ I(ni/100) + meta', scale='~ I(ni/100) + imag', data=metafopy.datasets.load('dat.bangertdrowns2004'))
    draws(record, 'seed', metafopy.vif(bangert, sim=50, seed=42, progbar=False))
    metafopy.set_seed(1234)
    draws(record, 'stream', metafopy.vif(bangert, sim=50, progbar=False))
    return record


def measure_permutest_ls_boundary() -> Record:
    record = Record()
    ref = test_audit4_numerics.REF['permutest']
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ alloc', data=test_audit4_numerics.bcg())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pt: typing.Any = metafopy.permutest(res, iter=50, seed=1)
    z, a = pt.zval_alpha_perm.to_numpy(), pt.alpha_perm.to_numpy()
    record.put('pval_alpha', pt.pval_alpha)
    record.put('pval_alpha.r', ref['pval_alpha'])
    record.put('QSp', pt.QSp)
    record.put('QSp.r', ref['QSp'])
    record.put('undefined_permutations', numpy.flatnonzero(numpy.isnan(z).any(axis=1)))
    record.put('boundary_permutations', numpy.flatnonzero((numpy.abs(a) >= 10).any(axis=1)))
    record.put('alpha_perm_max_abs', numpy.nanmax(numpy.abs(a), axis=1))
    return record


def measure_selmodel_print_halfnorm() -> Record:
    record = Record()
    case = 'hack_halfnorm_less'
    res = test_selmodel.fit(case)
    for name in ('beta', 'tau2', 'delta', 'se', 'ci_lb', 'ci_ub', 'll'):
        record.put(name, getattr(res, name))
    record.put('vb', res.vb)
    record.put('vd', res.vd)
    record.say(f'print: {str(res)!r}'[:2000])
    record.say(f'reference print: {test_selmodel.REF["fits"][case]["print"]!r}'[:2000])
    return record


# auditme: accept[non-alphabetical-dict-keys] the order is that of the spec's Failure groups table
GROUPS: dict[str, collections.abc.Callable[[], Record]] = {
    'ls-bcg-reference': measure_ls_bcg_reference,
    'ls-identity-alpha': measure_ls_identity_alpha,
    'selmodel-decreasing': measure_selmodel_decreasing,
    'glmm-um-rs-score': measure_glmm_um_rs_score,
    'cm-el-tight-se': measure_cm_el_tight_se,
    'mv-reml-score': measure_mv_reml_score,
    'stepcon-rau-mods': measure_stepcon_rau_mods,
    'ls-vif-sim': measure_ls_vif_sim,
    'permutest-ls-boundary': measure_permutest_ls_boundary,
    'ls-identity-maximum': measure_ls_identity_maximum,
    'selmodel-print-halfnorm': measure_selmodel_print_halfnorm,
}


def measure_groups(groups: collections.abc.Sequence[str] | None = None) -> dict[str, GroupMeasurement]:
    """Measure each group (all when ``None``) with the same helpers its tests call (``correctness.formulas``, ``correctness.glmm_formulas``, the test
    modules' fixtures); a group that raises is recorded with its error and the others still run.

    Raises:
        KeyError: a name in ``groups`` is not a group.
    """
    names = list(GROUPS) if groups is None else list(groups)
    unknown = [name for name in names if name not in GROUPS]
    if unknown:
        raise KeyError(f'not a group: {", ".join(unknown)}')
    out: dict[str, GroupMeasurement] = {}
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                record = GROUPS[name]()
            out[name] = GroupMeasurement(name, record.values, record.messages, None)
        # auditme: accept[silent-broad-except] a group that raises is recorded in its GroupMeasurement and the others run
        except Exception as error:
            # auditme: accept[except-interpolation] the error text is the record
            out[name] = GroupMeasurement(name, {}, [], f'{type(error).__name__}: {re.sub(r"\s+", " ", str(error))}')
    return out


def main(argv: list[str] | None = None) -> int:
    """``platform-probe.py [--group NAME ...] --out PATH``: write ``{'environment': ..., 'groups': ...}`` as JSON; exit 0, or 1 if any group raised."""
    parser = argparse.ArgumentParser(
        prog='platform-probe.py',
        description='Measure the numerical failure groups of the cross-platform numerics spec in this environment.',
    )
    parser.add_argument(
        '--group',
        action='append',
        choices=sorted(GROUPS),
        help='Group to measure (repeat for several); all groups by default',
    )
    parser.add_argument(
        '--out',
        required=True,
        type=pathlib.Path,
        help='JSON output path, outside the repository',
    )
    args = parser.parse_args(argv)
    measured = measure_groups(args.group)
    document = {'environment': environment(), 'groups': {name: dataclasses.asdict(m) for name, m in measured.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    failed = [name for name, m in measured.items() if m.error is not None]
    for name in failed:
        print(f'{name}: {measured[name].error}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
