"""The scope gate of the correctness suite and the checks of the harness itself.

The gate names every item of the core that has no ``published`` or ``derived`` case.
"""

import ast
import collections.abc
import dataclasses
import importlib
import inspect
import math
import pathlib
import types
import typing

import numpy
import pytest

import correctness.cases
import correctness.formulas
import correctness.glmm_formulas
import correctness.test_published
import metafopy
import metafopy.core.clubsandwich
import metafopy.core.glm
import metafopy.datasets
import metafopy.effects.measures
import metafopy.errors
import metafopy.methods.confint
import metafopy.methods.ls_resid
import metafopy.methods.robust
import metafopy.models.glmm_fit
import metafopy.models.mv_structs
import metafopy.models.rma_glmm
import metafopy.models.rma_ls_fit
import metafopy.models.rma_mv
import metafopy.models.rma_uni
import metafopy.models.sel_spec
import metafopy.models.tau2

HETEROGENEITY = ('H2', 'I2', 'QE', 'QM')
LS_HETEROGENEITY = ('H2', 'I2', 'QE')
LS_PREFIXES = ('ls.',)
BIAS_PREFIXES = ('fsn.', 'hc.', 'ranktest.', 'regtest.', 'selmodel.', 'tes.', 'trimfill.')
LS_PREDICT = ('both', 'fitted', 'newmods', 'newscale')
MV_DELETIONS = ('cluster', 'fixed', 'obs')
MV_DIAGNOSTICS = ('cooks_distance', 'dfbetas', 'rstudent')
UNI_FITS = ('ee', 'me', 're', 'tau2_fixed')
UNI_INFLUENCE = ('cov_r', 'dffits', 'hat', 'qe_del', 'tau2_del')
UNI_EXEMPT = ('knha', 'unweighted', 'weights')
LS_ANOVA = ('X', 'Z', 'att', 'btt', 'lrt.ML', 'lrt.REML')
LS_VIF = ('location', 'reestimate', 'scale')
MH_MEASURES = ('IRD', 'IRR', 'OR', 'RD', 'RR')
FORMULA_FILES = tuple(pathlib.Path(__file__).parent / f'{module}.py' for module in correctness.cases.FORMULA_MODULES)

# every argument escalc takes from the data, as one small dataset; escalc ignores the ones a measure does not use
BASE_INPUTS: dict[str, typing.Any] = {
    'ai': [12, 20, 8],
    'bi': [88, 80, 92],
    'ci': [18, 25, 14],
    'di': [82, 75, 86],
    'm1i': [10.0, 12.0, 11.0],
    'm2i': [8.0, 9.0, 10.0],
    'maxi': [9.0, 10.0, 11.0],
    'mi': [10.0, 12.0, 11.0],
    'mini': [1.0, 2.0, 3.0],
    'n1i': [100, 100, 100],
    'n2i': [100, 100, 100],
    'ni': [100, 100, 100],
    'pi': [0.3, 0.4, 0.5],
    'r2i': [0.2, 0.3, 0.25],
    'ri': [0.3, 0.4, 0.5],
    'sd1i': [2.0, 3.0, 2.5],
    'sd2i': [2.2, 2.8, 2.4],
    'sdi': [2.0, 3.0, 2.5],
    't1i': [1000.0, 1200.0, 900.0],
    't2i': [1000.0, 1100.0, 950.0],
    'ti': [1000.0, 1200.0, 900.0],
    'x1i': [12, 20, 8],
    'x2i': [18, 25, 14],
    'xi': [12, 20, 8],
}
ALPHA = {'ai': [0.7, 0.8, 0.75]}
AUC = {'ai': [0.6, 0.7, 0.65], 'mi': [0.1, 0.2, 0.15]}
# the measures whose input has a constraint the shared dataset breaks (a count equals xi + mi; a proportion lies in [0, 1]), and the ones that take
# their variance type from a summary the dataset already gives
INPUT_OVERRIDES: dict[str, dict[str, typing.Any]] = {
    'ABT': ALPHA,
    'AHW': ALPHA,
    'ARAW': ALPHA,
    'AUC': AUC,
    'CLES': AUC,
    'PAS': {'ni': [22, 32, 19]},
    'PFT': {'ni': [22, 32, 19]},
    'PLN': {'ni': [22, 32, 19]},
    'PLO': {'ni': [22, 32, 19]},
    'PR': {'ni': [22, 32, 19]},
    'PRZ': {'ni': [22, 32, 19]},
}
# AUCN and CLESN derive their variance type from di, ti and pi when these are given: the proportion ai alone leaves the type free
DROP_INPUTS = {'AUCN': ('di', 'pi', 'ti'), 'CLESN': ('di', 'pi', 'ti')}
GENERIC = {'vi': [0.01, 0.02, 0.03], 'yi': [0.1, 0.2, 0.3]}


def escalc_inputs(measure: str) -> dict[str, typing.Any]:
    """The smallest dataset on which ``escalc`` can compute ``measure``."""
    if measure == 'GEN':
        return dict(GENERIC)
    inputs = {**BASE_INPUTS, **INPUT_OVERRIDES.get(measure, {}), **({'ai': [0.6, 0.7, 0.65]} if measure in DROP_INPUTS else {})}
    for name in DROP_INPUTS.get(measure, ()):
        del inputs[name]
    return inputs


VTYPE_PAIRS = [(entry.measure, vtype) for entry in correctness.cases.load_vtypes() for vtype in entry.vtype]


def coverage_items() -> list[str]:
    """Every item of the core that needs a ``published`` or ``derived`` case, read from the code where the code lists the options."""
    items = [f'escalc.yi.{m}' for m in metafopy.effects.measures.MEASURES]
    items += [f'escalc.vi.{m}.{vtype}' for m, vtype in VTYPE_PAIRS]
    items += [f'tau2.{method}' for method in metafopy.models.tau2.ESTIMATORS]
    items += [f'test.{test}' for test in typing.get_args(metafopy.models.rma_uni.Test.__value__)]
    items += [f'confint.{kind}' for kind in typing.get_args(metafopy.methods.confint.CIType)]
    items += ['predict.pi']
    items += [f'heterogeneity.{statistic}' for statistic in HETEROGENEITY]
    items += [f'rma_mh.{m}.{part}' for m in MH_MEASURES for part in ('estimate', 'variance')]
    items += ['rma_peto']
    items += mv_robust_items()
    items += mv_diagnostics_items()
    items += uni_diagnostics_items()
    items += glmm_items()
    items += ls_items()
    items += uni_weights_items()
    items += bias_items()
    return items


def literal_options(func: collections.abc.Callable[..., typing.Any], parameter: str) -> tuple[str, ...]:
    """The string values of the ``Literal`` in ``parameter``'s annotation, through a ``type`` alias and ``| None``.

    Any other annotation raises ``TypeError``.
    """
    name = getattr(func, '__name__', repr(func))
    annotation = inspect.signature(func).parameters[parameter].annotation
    parts = typing.get_args(annotation) if isinstance(annotation, types.UnionType) else (annotation,)
    values: list[str] = []
    for part in parts:
        if part is type(None):
            continue
        if isinstance(part, typing.TypeAliasType):
            part = part.__value__
        if typing.get_origin(part) is not typing.Literal:
            raise TypeError(f'{name}: parameter {parameter!r} is annotated {annotation!r}, not a Literal of strings')
        values += [value for value in typing.get_args(part) if isinstance(value, str)]
    if not values:
        raise TypeError(f'{name}: parameter {parameter!r} has no Literal string values in {annotation!r}')
    return tuple(values)


def bias_items() -> list[str]:
    """The publication-bias items, each option list read from the code where the code lists it."""
    selmodel = importlib.import_module('metafopy.models.selmodel')
    models = literal_options(metafopy.regtest, 'model')
    predictors = literal_options(metafopy.regtest, 'predictor')
    items = [f'regtest.{model}.{predictor}' for model in models for predictor in predictors]
    items += ['regtest.limit', 'ranktest.tau', 'ranktest.pval.exact', 'ranktest.pval.normal']
    items += [f'trimfill.{statistic}.{estimator}' for statistic in ('k0', 'se_k0') for estimator in literal_options(metafopy.trimfill, 'estimator')]
    items += ['trimfill.p_k0', 'trimfill.side', 'trimfill.fill']
    items += [f'fsn.{kind}' for kind in literal_options(metafopy.fsn, 'type')]
    items += [f'fsn.pool.{pool}' for pool in literal_options(metafopy.fsn, 'pool')]
    items += [f'tes.test.{test}' for test in literal_options(metafopy.tes, 'test')]
    items += ['tes.E', 'tes.tau2', 'tes.theta_lim', 'hc.estimate', 'hc.ci']
    items += [f'selmodel.{kind}' for kind in metafopy.models.sel_spec.TYPE_OPTIONS]
    items += [f'selmodel.alternative.{a}' for a in selmodel.ALTERNATIVES]
    items += [f'selmodel.prec.{p}' for p in selmodel.PRECISIONS]
    items += ['selmodel.fixed_delta', 'selmodel.decreasing', 'selmodel.LRT']
    return items


def ls_items() -> list[str]:
    """The location-scale items, each option list read from the code where the code lists it."""
    links = typing.get_args(inspect.signature(metafopy.rma_uni).parameters['link'].annotation)
    items = [f'ls.fit.{method}.{link}' for method in metafopy.models.rma_ls_fit.LS_METHODS for link in links]
    items += [f'ls.test.{test}' for test in typing.get_args(metafopy.models.rma_uni.Test.__value__)]
    items += ['ls.optbeta', 'ls.scale.vcov', 'ls.scale.QS', 'ls.fitstats.ML', 'ls.fitstats.REML']
    items += [f'ls.heterogeneity.{statistic}' for statistic in LS_HETEROGENEITY]
    items += ['ls.confint.scale', 'ls.confint.location']
    items += [f'ls.predict.{situation}' for situation in LS_PREDICT]
    items += [f'ls.anova.{form}' for form in LS_ANOVA]
    items += ['ls.permutest.location.exact', 'ls.permutest.scale.exact']
    items += [f'ls.profile.{link}' for link in links]
    items += [f'ls.vif.{kind}' for kind in LS_VIF]
    items += ['ls.blup', 'ls.ranef']
    rstandard_types = typing.get_args(inspect.signature(metafopy.methods.ls_resid.rstandard_rma_ls).parameters['type'].annotation)
    items += [f'ls.rstandard.{kind}' for kind in rstandard_types]
    items += ['ls.unweighted', 'ls.alpha.fixed', 'ls.alpha.bounds', 'ls.beta.fixed.ML', 'ls.optbeta.REML', 'ls.optbeta.vb.ML']
    items += ['ls.fitstats.deviance', 'ls.unweighted.fitstats', 'ls.beta.fixed.REML', 'ls.optbeta.vb.REML']
    items += ['ls.weights', 'ls.weights.fitstats', 'ls.weights.knha']
    return items


def uni_weights_items() -> list[str]:
    """The user-weights items of a plain ``rma_uni`` fit: the estimates, then the fit statistics and ``knha``, which are exempt."""
    return ['rma_uni.weights', 'rma_uni.weights.fitstats', 'rma_uni.weights.knha']


def glmm_items() -> list[str]:
    """The ``rma_glmm`` items: each option list read from the code, the literals inside ``check_arguments`` and ``i2_h2`` listed here."""
    glmm = metafopy.models.rma_glmm
    items: list[str] = []
    for measure in (*glmm.SUPPORTED, *glmm.EXPERIMENTAL):
        if measure in metafopy.models.glmm_fit.ONE_GROUP:
            models: tuple[str, ...] = ('none',)
        elif measure in glmm.UNCONDITIONAL_ONLY:
            models = ('UM.FS', 'UM.RS')
        else:
            models = glmm.MODELS
        items += [f'rma_glmm.fit.{measure}.{model}.{method}' for model in models for method in correctness.glmm_formulas.METHODS]
    items += [f'rma_glmm.method.{m}' for m in glmm.METHODS]
    items += [f'rma_glmm.link.{link}' for link in metafopy.core.glm.LINKS]
    items += [f'rma_glmm.package.{p}' for p in metafopy.models.glmm_fit.PACKAGES]
    items += [f'rma_glmm.optimizer.{o}' for o in metafopy.models.glmm_fit.OPTIMIZERS]
    items += ['rma_glmm.test.z', 'rma_glmm.test.t']
    items += [f'rma_glmm.coding.{c}' for c in ('0.5', '1', '0')]
    items += ['rma_glmm.cor']
    items += [f'rma_glmm.heterogeneity.{s}' for s in ('QE_Wld', 'QE_LRT', 'QM', 'I2', 'H2')]
    # auditme: accept[literal-duplication] the values are the literals inside i2_h2, listed again here by the spec's design
    items += [f'rma_glmm.i2def.{d}' for d in ('1', '2')]
    items += ['rma_glmm.tau2_ci', 'rma_glmm.predict.pi']
    return items


def mv_robust_items() -> list[str]:
    """The rma_mv and robust items, each option list read from the code."""
    items = [f'rma_mv.method.{m}' for m in metafopy.models.rma_mv.METHODS]
    items += [f'rma_mv.struct.{s}' for s in metafopy.models.mv_structs.STRUCTS]
    items += [f'rma_mv.Rscale.{r}' for r in metafopy.models.rma_mv.RSCALES]
    items += [f'rma_mv.test.{t}' for t in metafopy.models.rma_mv.TESTS]
    items += [f'rma_mv.dfs.{d}' for d in metafopy.models.rma_mv.DFS]
    items += ['rma_mv.random.sigma2', 'rma_mv.random.nested', 'rma_mv.random.two_terms', 'rma_mv.random.R', 'rma_mv.weights']
    items += ['rma_mv.QE', 'rma_mv.QM', 'rma_mv.confint.pl', 'rma_mv.confint.fixed', 'rma_mv.predict.pi']
    items += ['robust.rma_uni', 'robust.rma_mv']
    items += [f'robust.adjust.{a}' for a in metafopy.methods.robust.ADJUST]
    items += [f'robust.vcov.{v}' for v in metafopy.core.clubsandwich.CR_TYPES]
    items += [f'robust.coef_test.{t}' for t in metafopy.methods.robust.COEF_TESTS]
    items += [f'robust.conf_test.{t}' for t in metafopy.methods.robust.CONF_TESTS]
    items += [f'robust.wald_test.{w}' for w in metafopy.core.clubsandwich.WALD_TESTS]
    return items


def mv_diagnostics_items() -> list[str]:
    """The ``rma_mv`` diagnostics items: each diagnostic by the deletion it makes (``obs``: each row with ``reestimate=True``, ``cluster``: whole
    clusters, ``fixed``: ``reestimate=False``), ``btt`` of Cook's distance, and ``ranef`` by term type and interval. The two constants name what no
    list in the code does."""
    items = [f'rma_mv.{name}.{deletion}' for name in MV_DIAGNOSTICS for deletion in MV_DELETIONS]
    items += ['rma_mv.cooks_distance.btt']
    items += [f'rma_mv.ranef.{kind}' for kind in ('G', 'GEN', 'H', 'R', 'pi.t', 'pi.z', 'sigma2')]
    return items


def uni_diagnostics_items() -> list[str]:
    """The ``rma_uni`` diagnostics items: each diagnostic by the kind of fit (``re``: estimated ``tau2``, no moderators; ``me``: with moderators; ``ee``:
    equal effects; ``tau2_fixed``: ``tau2`` given), ``btt`` of Cook's distance, the ``influence`` columns, the marginal ``rstandard``, ``ranef`` by fit
    and interval, and ``blup``; then the items the paper does not define (``knha``, ``unweighted``, ``weights``), which are exempt."""
    items = [f'rma_uni.{name}.{fit}' for name in MV_DIAGNOSTICS for fit in UNI_FITS]
    items += ['rma_uni.cooks_distance.btt']
    items += [f'rma_uni.ranef.{kind}' for kind in ('re', 'me', 'ee', 'pi.z', 'pi.t')]
    items += [f'rma_uni.influence.{column}' for column in UNI_INFLUENCE]
    items += ['rma_uni.rstandard.marginal', 'rma_uni.blup.re', 'rma_uni.blup.me']
    items += [f'rma_uni.{name}.{kind}' for name in MV_DIAGNOSTICS for kind in UNI_EXEMPT]
    return items


def missing_items(cases: list[correctness.cases.Case], exemptions: list[correctness.cases.Exemption], items: list[str] | None = None) -> list[str]:
    """The items with no ``published`` or ``derived`` case (a ``pass`` or ``mismatch`` one) and no exemption."""
    covered = {item for c in cases if c.kind in correctness.cases.GATE_KINDS for item in c.covers}
    exempt = {e.item for e in exemptions}
    return [item for item in (coverage_items() if items is None else items) if item not in covered and item not in exempt]


def test_case_files_are_valid() -> None:
    cases = correctness.cases.load_cases()
    assert len({c.id for c in cases}) == len(cases)
    correctness.cases.load_exemptions()


def test_covers_and_exemptions_name_known_items() -> None:
    known = set(coverage_items())
    named = [item for c in correctness.cases.load_cases() for item in c.covers]
    named += [e.item for e in correctness.cases.load_exemptions()]
    assert sorted(set(named) - known) == []


def metafopy_imports(source: str) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or '')
    return [m for m in modules if m.split('.')[0] == 'metafopy']


@pytest.mark.parametrize('path', FORMULA_FILES, ids=[path.name for path in FORMULA_FILES])
def test_formulas_import_nothing_from_metafopy(path: pathlib.Path) -> None:
    assert metafopy_imports(path.read_text(encoding='utf-8')) == []


def test_a_formula_module_that_imports_metafopy_is_caught() -> None:
    assert metafopy_imports('import metafopy.models.rma_glmm\nimport numpy\n') == ['metafopy.models.rma_glmm']
    assert metafopy_imports('from metafopy import escalc\n') == ['metafopy']
    assert metafopy_imports('import numpy\nimport scipy.stats\n') == []


def test_each_unsupported_optimizer_exemption_still_raises() -> None:
    prefix = 'rma_glmm.optimizer.'
    stale: list[str] = []
    for exemption in correctness.cases.load_exemptions():
        if not exemption.item.startswith(prefix) or 'NotSupportedError' not in exemption.reason:
            continue
        try:
            metafopy.rma_glmm(
                ai='ai',
                n1i='n1i',
                ci='ci',
                n2i='n2i',
                data=metafopy.datasets.load('dat.nielweise2007'),
                measure='OR',
                model='CM.EL',
                method='ML',
                control={'optimizer': exemption.item.removeprefix(prefix)},
            )
        except metafopy.errors.NotSupportedError:
            continue
        stale.append(exemption.item)
    assert not stale, f'exempted as NotSupportedError but no longer raising it; replace the exemption with a case: {", ".join(stale)}'


def test_vtype_table_lists_every_measure_once() -> None:
    measures = [entry.measure for entry in correctness.cases.load_vtypes()]
    assert sorted(measures) == sorted(metafopy.effects.measures.MEASURES)
    assert all(vtype in metafopy.effects.measures.VTYPES for _, vtype in VTYPE_PAIRS)


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize(('measure', 'vtype'), VTYPE_PAIRS, ids=[f'{m}-{v}' for m, v in VTYPE_PAIRS])
def test_escalc_accepts_each_listed_vtype(measure: str, vtype: str) -> None:
    out = metafopy.escalc(measure, vtype=vtype, **escalc_inputs(measure))
    assert len(out['yi']) == len(out['vi']) == 3


def derived_missing(
    cases: list[correctness.cases.Case],
    exemptions: list[correctness.cases.Exemption],
    prefixes: list[str],
    items: list[str] | None = None,
) -> list[str]:
    """The items under a ``derived_required`` prefix with no ``derived`` case and no exemption."""
    covered = {item for c in cases if c.kind == 'derived' for item in c.covers}
    exempt = {e.item for e in exemptions}
    return [item for item in (coverage_items() if items is None else items) if item.startswith(tuple(prefixes)) and item not in covered and item not in exempt]


def test_every_location_scale_item_has_a_derived_case() -> None:
    missing = derived_missing(correctness.cases.load_cases(), correctness.cases.load_exemptions(), correctness.cases.load_derived_required())
    assert not missing, f'{len(missing)} items have no derived case: {", ".join(missing)}'


def test_a_derived_required_item_needs_a_derived_case_or_an_exemption() -> None:
    (published,) = correctness.cases.parse_cases(VALID, 'inline.toml')
    x_item, y_item, other = 'ls.x', 'ls.y', 'other.z'
    prefixes = list(LS_PREFIXES)
    published_only = dataclasses.replace(published, covers=(x_item,))
    derived = valid_case(id='unit.derived', kind='derived', source=None, covers=(x_item,))
    exempt = [correctness.cases.Exemption(item=y_item, reason='nothing to derive', file='inline.toml')]
    items = [x_item, y_item, other]
    assert derived_missing([published_only], [], prefixes, items) == [x_item, y_item]
    assert derived_missing([published_only, derived], [], prefixes, items) == [y_item]
    assert derived_missing([published_only], exempt, prefixes, items) == [x_item]


def test_an_invalid_derived_required_raises_naming_the_file(tmp_path: pathlib.Path) -> None:
    (tmp_path / 'bad.toml').write_text('derived_required = "ls."\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'bad\.toml'):
        correctness.cases.load_derived_required(tmp_path)


def test_derived_required_is_read_from_every_file(tmp_path: pathlib.Path) -> None:
    prefixes = list(LS_PREFIXES)
    (tmp_path / 'a.toml').write_text(f'derived_required = {prefixes!r}\n'.replace("'", '"'), encoding='utf-8')
    (tmp_path / 'b.toml').write_text('', encoding='utf-8')
    assert correctness.cases.load_derived_required(tmp_path) == prefixes


def test_every_core_item_has_a_case() -> None:
    items = [item for item in coverage_items() if not item.startswith(BIAS_PREFIXES)]
    missing = missing_items(correctness.cases.load_cases(), correctness.cases.load_exemptions(), items)
    assert not missing, f'{len(missing)} items have no published or derived case: {", ".join(missing)}'


def test_every_bias_item_has_a_case() -> None:
    missing = missing_items(correctness.cases.load_cases(), correctness.cases.load_exemptions(), bias_items())
    assert not missing, f'{len(missing)} items have no published or derived case: {", ".join(missing)}'


# ---- the harness itself, on inline case files

VALID = """
[[case]]
id = "unit.md"
kind = "published"
function = "escalc"
args = { measure = "MD", m1i = "m1i", sd1i = "sd1i", n1i = "n1i", m2i = "m2i", sd2i = "sd2i", n2i = "n2i" }
expect = { "yi[0]" = "2.0", "vi[0]" = "0.5" }
covers = ["escalc.yi.MD"]

[case.source]
citation = "A unit test"
locator = "none"
read = 2026-11-11

[case.data]
m1i = [10.0]
m2i = [8.0]
sd1i = [2.0]
sd2i = [3.0]
n1i = [20.0]
n2i = [30.0]
"""


def valid_case(**replace: typing.Any) -> correctness.cases.Case:
    return dataclasses.replace(correctness.cases.parse_cases(VALID, 'inline.toml')[0], **replace)


@pytest.mark.parametrize(
    ('expected', 'tolerance'),
    [('0.0193', 5e-05), ('684', 0.5), ('-15.106', 5e-04), ('1.5e-3', 5e-05), ('2.0', 0.05)],
)
def test_published_tolerance_is_half_the_last_digit(expected: str, tolerance: float) -> None:
    assert correctness.cases.published_tolerance(expected) == tolerance


def test_published_tolerance_refuses_a_non_number() -> None:
    with pytest.raises(ValueError, match='not a printed number'):
        correctness.cases.published_tolerance('abc')


def test_a_valid_published_case_runs_and_passes() -> None:
    (case,) = correctness.cases.parse_cases(VALID, 'inline.toml')
    assert case.source is not None
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def test_a_value_outside_the_printed_digit_fails_naming_case_key_and_source() -> None:
    case = valid_case(expect={'yi[0]': '2.2'})
    with pytest.raises(AssertionError, match=r'unit\.md: yi\[0\].*2\.2.*A unit test, none'):
        correctness.cases.check_case(case, correctness.cases.run_case(case))


def test_check_printed_passes_inside_and_fails_outside_the_printed_tolerance() -> None:
    case = valid_case(printed={'x': '2.0'})
    correctness.cases.check_printed(case, 'x', 2.049)
    with pytest.raises(AssertionError, match=r'unit\.md: x: metafopy gives 2\.06.*2\.0.*A unit test, none'):
        correctness.cases.check_printed(case, 'x', 2.06)
    with pytest.raises(LookupError, match='no printed value'):
        correctness.cases.check_printed(case, 'y', 2.0)


def test_a_published_check_refuses_a_printed_name_it_does_not_know() -> None:
    (case,) = correctness.cases.parse_cases(
        broken('covers = ["escalc.yi.MD"]', 'check = "test_published.back_transformed"\nprinted = { exp_beta = "1.0", nonsense = "1.0" }'),
        'inline.toml',
    )

    fit = types.SimpleNamespace(beta=[0.0], ci_lb=[0.0], ci_ub=[0.0])
    with pytest.raises(LookupError, match='nonsense'):
        correctness.test_published.back_transformed(case, fit)


def test_a_published_case_with_printed_values_alone_loads_and_with_neither_is_refused() -> None:
    (case,) = correctness.cases.parse_cases(
        broken('expect = { "yi[0]" = "2.0", "vi[0]" = "0.5" }', 'check = "test_published.back_transformed"\nprinted = { exp_beta = "1.0" }'),
        'inline.toml',
    )
    assert case.printed == {'exp_beta': '1.0'}
    assert not case.expect
    with pytest.raises(ValueError, match='needs expect or printed values'):
        correctness.cases.parse_cases(broken('expect = { "yi[0]" = "2.0", "vi[0]" = "0.5" }', 'expect = {}'), 'inline.toml')


def test_back_transformed_compares_exp_of_the_fit() -> None:
    (case,) = correctness.cases.parse_cases(
        broken('covers = ["escalc.yi.MD"]', 'check = "test_published.back_transformed"\nprinted = { exp_beta = "2.72", exp_ci_ub = "7.4" }'),
        'inline.toml',
    )

    fit = types.SimpleNamespace(beta=[1.0], ci_lb=[0.0], ci_ub=[2.0])
    correctness.test_published.back_transformed(case, fit)
    fit.beta = [1.5]
    with pytest.raises(AssertionError, match='exp_beta'):
        correctness.test_published.back_transformed(case, fit)


def test_a_derived_case_takes_its_value_from_a_formula() -> None:
    case = correctness.cases.parse_cases(
        """
[[case]]
id = "unit.rd"
kind = "derived"
function = "escalc"
args = { measure = "RD", ai = "ai", bi = "bi", ci = "ci", di = "di" }
expect = { yi = "formulas.risk_difference_yi" }
[case.data]
ai = [1.0, 2.0]
bi = [3.0, 2.0]
ci = [2.0, 1.0]
di = [2.0, 3.0]
""",
        'inline.toml',
    )[0]
    correctness.cases.check_case(case, correctness.cases.run_case(case))
    wrong = dataclasses.replace(case, expect={'yi': 'formulas.risk_difference_vi'})
    with pytest.raises(AssertionError, match=r'unit\.rd: yi'):
        correctness.cases.check_case(wrong, correctness.cases.run_case(wrong))


def test_an_unknown_function_or_attribute_names_what_it_could_not_resolve() -> None:
    with pytest.raises(LookupError, match='no_such_function'):
        correctness.cases.run_case(valid_case(function='no_such_function'))
    case = valid_case(expect={'no_such_attribute': '1'})
    with pytest.raises(LookupError, match='no_such_attribute'):
        correctness.cases.check_case(case, correctness.cases.run_case(case))
    with pytest.raises(LookupError, match='no_such_formula'):
        correctness.cases.run_case(valid_case(function='formulas.no_such_formula'))


def test_a_glmm_formula_resolves_and_an_unknown_module_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(correctness.glmm_formulas, 'probe', lambda: 1.0, raising=False)
    func = correctness.cases.resolve_formula('glmm_formulas.probe', 'unit')
    assert func() == 1.0
    with pytest.raises(LookupError, match=r'glmm_formulas\.py has no function'):
        correctness.cases.resolve_formula('glmm_formulas.no_such_formula', 'unit')
    with pytest.raises(ValueError, match=r'inline\.toml: case unit\.bad: function .other_formulas\.f. must be'):
        correctness.cases.parse_cases('[[case]]\nid = "unit.bad"\nkind = "derived"\nfunction = "other_formulas.f"\nexpect = { yi = 1.0 }\n', 'inline.toml')


def test_a_mismatch_case_becomes_a_strict_xfail() -> None:
    param = correctness.cases.case_param(valid_case(status='mismatch', reason='source 2.0, metafopy 2.2'))
    (mark,) = param.marks
    assert mark.name == 'xfail'
    assert mark.kwargs == {'strict': True, 'reason': 'source 2.0, metafopy 2.2'}
    assert list(correctness.cases.case_param(valid_case()).marks) == []


def broken(old: str, new: str) -> str:
    assert old in VALID
    return VALID.replace(old, new, 1)


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        (broken('id = "unit.md"', 'id = "unit md"'), r'inline\.toml.*invalid id'),
        (broken('id = "unit.md"\n', ''), r'inline\.toml.*invalid id'),
        (broken('kind = "published"', 'kind = "guess"'), r'inline\.toml: case unit\.md: unknown kind'),
        (broken('function = "escalc"\n', ''), r"inline\.toml: case unit\.md: missing 'function'"),
        (broken('function = "escalc"', 'function = "metafopy.escalc"'), r'case unit\.md: function'),
        (broken('"vi[0]" = "0.5"', '"vi[0]" = 0.5'), r'case unit\.md: published value of .vi\[0\]. must be a string'),
        (broken('"yi[0]" = "2.0"', '"yi[0]" = "two"'), r'not a printed number'),
        (broken('[case.source]\ncitation = "A unit test"\nlocator = "none"\nread = 2026-11-11\n', ''), r'case unit\.md: a published case needs a source'),
        (broken('read = 2026-11-11', 'read = "yesterday"'), r'source\.read .yesterday. is not a date'),
        (broken('covers = ["escalc.yi.MD"]', 'status = "mismatch"'), r'case unit\.md: a mismatch case needs a reason'),
        (broken('covers = ["escalc.yi.MD"]', 'colour = "red"'), r"case unit\.md: unknown key\(s\) \['colour'\]"),
        (broken('covers = ["escalc.yi.MD"]', 'check = "properties.f"'), r'case unit\.md: check'),
        (broken('covers = ["escalc.yi.MD"]', 'dataset = "dat.bcg"'), r'a dataset or inline data, not both'),
        (broken('covers = ["escalc.yi.MD"]', 'tolerance = -1'), r'tolerance must be a positive number'),
        (broken('covers = ["escalc.yi.MD"]', 'printed = { x = "1.0" }'), r'inline\.toml: case unit\.md: printed is allowed only with'),
        (
            broken('kind = "published"', 'kind = "derived"')
            .replace('"yi[0]" = "2.0", "vi[0]" = "0.5"', '"yi[0]" = 2.0, "vi[0]" = 0.5')
            .replace('covers = ["escalc.yi.MD"]', 'check = "test_published.back_transformed"\nprinted = { x = "1.0" }'),
            r'inline\.toml: case unit\.md: printed is allowed only with',
        ),
        (
            broken('covers = ["escalc.yi.MD"]', 'check = "test_published.back_transformed"\nprinted = { x = "one" }'),
            r'inline\.toml: case unit\.md: printed value of .x. \(.one.\) is not a printed number',
        ),
        (
            broken('covers = ["escalc.yi.MD"]', 'check = "test_published.back_transformed"\nprinted = { x = 1.0 }'),
            r'inline\.toml: case unit\.md: printed value of .x. must be a string',
        ),
        (broken('covers = ["escalc.yi.MD"]', 'bounds = ["alpha.0.lower"]'), r'case unit\.md: bounds must be a non-empty list'),
        (broken('covers = ["escalc.yi.MD"]', 'bounds = []'), r'case unit\.md: bounds must be a non-empty list'),
        (VALID + VALID, r"inline\.toml: duplicate case id 'unit\.md'"),
        ('[[case]\nid = 1', r'inline\.toml: invalid TOML'),
        ('[mystery]\nx = 1', r'inline\.toml: unknown top-level key'),
    ],
)
def test_an_invalid_case_file_raises_naming_file_and_case(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        correctness.cases.parse_cases(text, 'inline.toml')


def test_a_case_names_the_bounds_it_checks() -> None:
    (case,) = correctness.cases.parse_cases(broken('covers = ["escalc.yi.MD"]', 'bounds = ["alpha.1.lower", "alpha.2.upper"]'), 'inline.toml')
    assert case.bounds == ('alpha.1.lower', 'alpha.2.upper')
    assert valid_case().bounds is None


def test_a_duplicate_id_across_files_names_the_second_file(tmp_path: pathlib.Path) -> None:
    (tmp_path / 'a.toml').write_text(VALID, encoding='utf-8')
    (tmp_path / 'b.toml').write_text(VALID, encoding='utf-8')
    with pytest.raises(ValueError, match=r"b\.toml: duplicate case id 'unit\.md', also in a\.toml"):
        correctness.cases.load_cases(directory=tmp_path)


def test_an_invalid_file_fails_the_load_naming_it(tmp_path: pathlib.Path) -> None:
    (tmp_path / 'broken.toml').write_text(broken('kind = "published"', 'kind = "guess"'), encoding='utf-8')
    with pytest.raises(ValueError, match=r'broken\.toml'):
        correctness.cases.load_cases(directory=tmp_path)


def test_the_gate_lists_an_item_until_a_published_or_derived_case_covers_it() -> None:
    (published,) = correctness.cases.parse_cases(VALID, 'inline.toml')
    prop = valid_case(id='unit.prop', kind='property', source=None, covers=('escalc.yi.RR',))
    items = ['escalc.yi.MD', 'escalc.yi.RR', 'escalc.yi.OR']
    md, rr, odds_ratio = items
    exempt = [correctness.cases.Exemption(item=odds_ratio, reason='no formula in the defining paper', file='inline.toml')]
    assert missing_items([prop], exempt, items) == [md, rr]
    assert missing_items([published, prop], exempt, items) == [rr]


# ---- the `then` follow-on call and the `vblocks` and `vconst` setups

THEN_CASE = """
[[case]]
id = "unit.then"
kind = "property"
function = "rma_uni"
args = { yi = "yi", vi = "vi", method = "FE" }
then = { function = "robust", args = { cluster = "district" } }
expect = { "beta[0]" = 0.0 }

[case.data]
yi = [0.1, 0.4, 0.2, 0.5]
vi = [0.02, 0.03, 0.025, 0.04]
district = [1.0, 1.0, 2.0, 2.0]
"""

VBLOCKS_CASE = """
[[case]]
id = "unit.vblocks"
kind = "property"
function = "rma_mv"
args = { yi = "yi", method = "FE" }
setup = { vblocks = { cluster = "trial", columns = ["v1i", "v2i"] } }
expect = { "beta[0]" = 0.0 }

[case.data]
yi = [0.1, 0.4, 0.2, 0.5]
trial = [1.0, 1.0, 2.0, 2.0]
v1i = [0.02, 0.01, 0.03, 0.005]
v2i = [0.01, 0.03, 0.005, 0.02]
"""

VCONST_CASE = """
[[case]]
id = "unit.vconst"
kind = "property"
function = "rma_mv"
args = { yi = "yi", method = "FE" }
setup = { vconst = { cluster = "study", vi = "vi", rho = 0.5 } }
expect = { "beta[0]" = 0.0 }

[case.data]
yi = [0.1, 0.4, 0.2]
study = [1.0, 1.0, 2.0]
vi = [0.04, 0.09, 0.16]
"""


# auditme: accept[non-alphabetical-literal] the order is the order of the test's parameters
TEXT_PATTERN = ('text', 'pattern')


def one_case(text: str) -> correctness.cases.Case:
    return correctness.cases.parse_cases(text, 'inline.toml')[0]


def test_then_calls_robust_on_the_fit() -> None:
    case = one_case(THEN_CASE)
    assert case.then == correctness.cases.Step(function='robust', args={'cluster': 'district'})
    plain = correctness.cases.run_case(dataclasses.replace(case, then=None))
    result = correctness.cases.run_case(case)
    assert type(result) is type(plain)
    assert result.robust is not None
    assert plain.robust is None


def test_vblocks_builds_the_block_diagonal_matrix() -> None:
    case = one_case(VBLOCKS_CASE)
    assert case.setup is not None
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    expected = numpy.array(
        [
            [0.02, 0.01, 0.0, 0.0],
            [0.01, 0.03, 0.0, 0.0],
            [0.0, 0.0, 0.03, 0.005],
            [0.0, 0.0, 0.005, 0.02],
        ]
    )
    actual = correctness.cases.build_v(frame, case.setup, 'inline.toml: case unit.vblocks')
    numpy.testing.assert_array_equal(actual, expected)


def test_vconst_builds_a_constant_within_cluster_correlation() -> None:
    case = one_case(VCONST_CASE)
    assert case.setup is not None
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    covariance = 0.5 * numpy.sqrt(0.04 * 0.09)
    expected = numpy.array([[0.04, covariance, 0.0], [covariance, 0.09, 0.0], [0.0, 0.0, 0.16]])
    numpy.testing.assert_allclose(correctness.cases.build_v(frame, case.setup, 'inline.toml: case unit.vconst'), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize('text', [VBLOCKS_CASE, VCONST_CASE])
def test_a_v_setup_is_passed_to_the_call_as_v(text: str) -> None:
    case = one_case(text)
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    assert case.setup is not None
    result = correctness.cases.run_case(case)
    assert result.k == len(correctness.cases.build_v(frame, case.setup, 'inline.toml'))


@pytest.mark.parametrize(
    TEXT_PATTERN,
    [
        (THEN_CASE.replace('then = { function', 'then = { colour = 1, function'), r'inline\.toml: case unit\.then: then: unknown key\(s\) \[.colour.\]'),
        (THEN_CASE.replace('function = "robust", ', ''), r'inline\.toml: case unit\.then: then: function None'),
        (
            VBLOCKS_CASE.replace('setup = {', 'setup = { vconst = { cluster = "trial", vi = "v1i", rho = 0.5 },'),
            r'inline\.toml: case unit\.vblocks: setup has both vblocks and vconst',
        ),
        (VBLOCKS_CASE.replace('cluster = "trial", ', ''), r'inline\.toml: case unit\.vblocks: setup\.vblocks has no cluster'),
        (VCONST_CASE.replace('cluster = "study", ', ''), r'inline\.toml: case unit\.vconst: setup\.vconst has no cluster'),
        (VCONST_CASE.replace(', rho = 0.5', ''), r'inline\.toml: case unit\.vconst: setup\.vconst takes exactly'),
    ],
)
def test_a_malformed_then_or_v_setup_raises_naming_file_and_case(text: str, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        correctness.cases.parse_cases(text, 'inline.toml')


def test_an_unknown_follow_on_function_raises_naming_it() -> None:
    case = dataclasses.replace(one_case(THEN_CASE), then=correctness.cases.Step(function='no_such_follow_on'))
    with pytest.raises(LookupError, match='no_such_follow_on'):
        correctness.cases.run_case(case)


def test_vblocks_reads_the_first_columns_for_a_smaller_cluster() -> None:
    """A cluster with fewer rows than ``columns`` names (a study missing an arm or an endpoint) takes its block from the first columns."""
    case = one_case(VBLOCKS_CASE)
    assert case.setup is not None
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    actual = correctness.cases.build_v(frame.iloc[:3], case.setup, 'inline.toml: case unit.vblocks')
    expected = numpy.array([[0.02, 0.01, 0.0], [0.01, 0.03, 0.0], [0.0, 0.0, 0.03]])
    numpy.testing.assert_array_equal(actual, expected)


def test_build_v_refuses_a_block_too_large_or_one_not_symmetric() -> None:
    case = one_case(VBLOCKS_CASE)
    assert case.setup is not None
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    where = 'inline.toml: case unit.vblocks'
    merged = frame.assign(trial=[1.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match=r'inline\.toml: case unit\.vblocks: cluster 1\.0 has 3 rows but vblocks names 2 columns'):
        correctness.cases.build_v(merged, case.setup, where)
    asymmetric = frame.copy()
    asymmetric.loc[1, 'v1i'] = 0.5
    with pytest.raises(ValueError, match=r'inline\.toml: case unit\.vblocks: the block of cluster 1\.0 is not symmetric'):
        correctness.cases.build_v(asymmetric, case.setup, where)


# ---- the `subset` and `tree_r` setups

SUBSET_CASE = """
[[case]]
id = "unit.subset"
kind = "property"
function = "rma_uni"
args = { yi = "yi", vi = "vi", method = "FE" }
setup = { subset = { column = "group", equals = "b" } }
expect = { "beta[0]" = 0.0 }

[case.data]
yi = [0.1, 0.4, 0.2, 0.5]
vi = [0.02, 0.03, 0.025, 0.04]
group = ["a", "b", "a", "b"]
"""

TREE_R_CASE = """
[[case]]
id = "unit.tree_r"
kind = "property"
function = "rma_mv"
args = { yi = "yi", V = "vi", random = "~ 1 | species.id", sigma2 = 0.01 }
setup = { tree_r = { tree = "moura2021", factor = "species.id" } }
expect = { "beta[0]" = 0.0 }

[case.data]
yi = [0.1, 0.4]
vi = [0.02, 0.03]
"species.id" = ["Gammarus_pulex_gallicus_ott1024016", "Trapezia_bidentata_ott787242"]
"""


def test_subset_keeps_the_rows_that_match() -> None:
    case = one_case(SUBSET_CASE)
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    assert frame['yi'].tolist() == [0.4, 0.5]
    assert frame.index.tolist() == [0, 1]
    assert correctness.cases.run_case(case).k == 2


def test_tree_r_is_the_shared_branch_length_of_the_tips() -> None:
    reference = metafopy.datasets.load_tree('moura2021').vcv()
    built = correctness.cases.load_tree_r('moura2021')
    assert list(built.index) == list(reference.index)
    numpy.testing.assert_allclose(built.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-12)
    assert correctness.cases.run_case(one_case(TREE_R_CASE)).k == 2


@pytest.mark.parametrize(
    TEXT_PATTERN,
    [
        (SUBSET_CASE.replace(', equals = "b"', ''), r'inline\.toml: case unit\.subset: setup\.subset takes exactly'),
        (TREE_R_CASE.replace(', factor = "species.id"', ''), r'inline\.toml: case unit\.tree_r: setup\.tree_r takes exactly'),
        (TREE_R_CASE.replace('tree = "moura2021"', 'tree = 3'), r'inline\.toml: case unit\.tree_r: setup\.tree_r takes exactly'),
    ],
)
def test_a_malformed_subset_or_tree_setup_raises_naming_file_and_case(text: str, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        correctness.cases.parse_cases(text, 'inline.toml')


# ---- the `where`, `drop` and `rma_uni` setups and `literal_options`

WHERE_CASE = """
[[case]]
id = "unit.where"
kind = "property"
function = "rma_uni"
args = { yi = "yi", vi = "vi", method = "FE" }
setup = { where = { group = "b" }, drop = [0] }
expect = { "beta[0]" = 0.0 }

[case.data]
yi = [0.1, 0.4, 0.2, 0.5, 0.3]
vi = [0.02, 0.03, 0.025, 0.04, 0.05]
group = ["a", "b", "a", "b", "b"]
"""

RMA_UNI_CASE = """
[[case]]
id = "unit.fit"
kind = "property"
function = "trimfill"
args = { estimator = "R0" }
setup = { rma_uni = { method = "FE" } }
expect = { k0 = 0.0 }
dataset = "dat.hackshaw1998"
"""


def test_where_then_drop_keeps_the_right_rows() -> None:
    frame = correctness.cases.case_frame(one_case(WHERE_CASE))
    assert frame is not None
    assert frame['yi'].tolist() == [0.5, 0.3]
    assert frame.index.tolist() == [0, 1]


def test_drop_alone_removes_the_named_rows() -> None:
    case = one_case(WHERE_CASE.replace('where = { group = "b" }, ', ''))
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    assert frame['yi'].tolist() == [0.4, 0.2, 0.5, 0.3]


@pytest.mark.parametrize(
    TEXT_PATTERN,
    [
        (WHERE_CASE.replace('setup = {', 'setup = { colour = {},'), r'inline\.toml: case unit\.where: setup must be a table whose keys'),
        (WHERE_CASE.replace('drop = [0]', 'drop = "x"'), r'inline\.toml: case unit\.where: setup\.drop must be a list'),
    ],
)
def test_a_malformed_where_or_drop_setup_raises_naming_file_and_case(text: str, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        correctness.cases.parse_cases(text, 'inline.toml')


@pytest.mark.parametrize(
    TEXT_PATTERN,
    [
        (WHERE_CASE.replace('group = "b"', 'colour = "b"'), r'inline\.toml: case unit\.where: setup\.where column .colour. is not in the data'),
        (WHERE_CASE.replace('drop = [0]', 'drop = [7]'), r'inline\.toml: case unit\.where: setup\.drop index \[7\] is outside the 3 rows'),
    ],
)
def test_a_where_or_drop_that_does_not_fit_the_data_raises_naming_file_and_case(text: str, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        correctness.cases.case_frame(one_case(text))


def test_setup_rma_uni_passes_the_fitted_model_and_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    case = one_case(RMA_UNI_CASE)
    seen: dict[str, typing.Any] = {}
    real = metafopy.trimfill

    def spy(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        seen['args'], seen['kwargs'] = args, kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(metafopy, 'trimfill', spy)
    result = correctness.cases.run_case(case)
    (fit,) = seen['args']
    assert isinstance(fit, metafopy.models.rma_uni.RMAResult)
    assert seen['kwargs'] == {'estimator': 'R0'}
    assert hasattr(result, 'k0')


type Colour = typing.Literal['cyan', 'magenta']
type Shade = typing.Literal['dark', 'light']


def probe(a: Colour, b: Shade | None, c: int) -> None:
    del a, b, c


def test_literal_options_reads_a_type_alias_and_an_optional_literal() -> None:
    assert literal_options(probe, 'a') == typing.get_args(Colour.__value__)
    assert literal_options(probe, 'b') == typing.get_args(Shade.__value__)
    assert 'fisher' in literal_options(metafopy.fsn, 'pool')
    predictor = importlib.import_module('metafopy.bias.regtest').Predictor
    assert literal_options(metafopy.regtest, 'predictor') == typing.get_args(predictor.__value__)


def test_literal_options_refuses_an_annotation_that_is_not_a_literal() -> None:
    with pytest.raises(TypeError, match=r"probe: parameter 'c'"):
        literal_options(probe, 'c')


MV_YI = numpy.array([0.3, -0.2, 0.5, 0.1, 0.8, -0.4, 0.6])
MV_VI = numpy.array([0.05, 0.08, 0.04, 0.06, 0.09, 0.07, 0.05])
MV_X = numpy.column_stack([numpy.ones(7), [1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.0]])
MV_TAU2 = 0.12


def sigma2_term(outer: list[int]) -> correctness.formulas.RandomTerm:
    codes = numpy.array(outer, dtype=numpy.intp)
    return correctness.formulas.RandomTerm(struct=correctness.formulas.SIGMA2, outer=codes, inner=numpy.zeros_like(codes), n_inner=1)


def test_mv_deletion_without_reestimation_matches_the_rma_uni_closed_forms() -> None:
    k = len(MV_YI)
    result = correctness.formulas.mv_deletion(
        MV_YI,
        numpy.diag(MV_VI),
        MV_X,
        [sigma2_term(list(range(k)))],
        'REML',
        [numpy.array([MV_TAU2])],
        [numpy.zeros(0)],
        [numpy.array([i]) for i in range(k)],
        False,
    )
    w = 1 / (MV_VI + MV_TAU2)
    vb = numpy.linalg.inv(MV_X.T @ (w[:, None] * MV_X))
    b = vb @ MV_X.T @ (w * MV_YI)
    for i in range(k):
        keep = numpy.arange(k) != i
        vb_i = numpy.linalg.inv(MV_X[keep].T @ (w[keep, None] * MV_X[keep]))
        b_i = vb_i @ MV_X[keep].T @ (w[keep] * MV_YI[keep])
        diff = b - b_i
        resid = MV_YI[i] - MV_X[i] @ b_i
        se = math.sqrt(MV_VI[i] + MV_TAU2 + MV_X[i] @ vb_i @ MV_X[i])
        assert result.cook[i] == pytest.approx(diff @ numpy.linalg.inv(vb) @ diff, abs=1e-10)
        assert result.dfbetas[i] == pytest.approx(diff / numpy.sqrt(numpy.diag(vb)), abs=1e-10)
        assert result.resid[i] == pytest.approx(resid, abs=1e-10)
        assert result.se[i] == pytest.approx(se, abs=1e-10)
        assert result.z[i] == pytest.approx(resid / se, abs=1e-10)
        assert result.X2[i] == pytest.approx((resid / se) ** 2, abs=1e-10)
    assert all(result.converged)


def test_uni_deletion_matches_the_intercept_only_closed_forms_and_refuses_a_rank_deficient_deletion() -> None:
    k = len(MV_YI)
    result = correctness.formulas.uni_deletion(MV_YI, MV_VI, 'REML', tau2=MV_TAU2)
    w = 1 / (MV_VI + MV_TAU2)
    mu = float(w @ MV_YI / w.sum())
    for i in range(k):
        keep = numpy.arange(k) != i
        mu_i = float(w[keep] @ MV_YI[keep] / w[keep].sum())
        assert result.dfbetas[i, 0] == pytest.approx((mu - mu_i) * math.sqrt(w.sum()), abs=1e-10)
        assert result.cook[i] == pytest.approx((mu - mu_i) ** 2 * w.sum(), abs=1e-10)
        assert result.resid[i] == pytest.approx(MV_YI[i] - mu_i, abs=1e-10)
        assert result.se[i] == pytest.approx(math.sqrt(MV_VI[i] + MV_TAU2 + 1 / w[keep].sum()), abs=1e-10)
        assert result.tau2_del[i] == MV_TAU2
        assert result.hat[i] == pytest.approx(w[i] / w.sum(), abs=1e-10)
    only = {'d': numpy.array([0.0] * 6 + [1.0])}
    with pytest.raises(ValueError, match='study 6'):
        correctness.formulas.uni_deletion(MV_YI, MV_VI, 'REML', mods='~ d', data=only, tau2=MV_TAU2)


def test_uni_blups_match_the_marginal_form() -> None:
    result = correctness.formulas.uni_blups(MV_YI, MV_VI, MV_X, MV_TAU2)
    w = 1 / (MV_VI + MV_TAU2)
    vb = numpy.linalg.inv(MV_X.T @ (w[:, None] * MV_X))
    b = vb @ MV_X.T @ (w * MV_YI)
    lam = MV_TAU2 / (MV_TAU2 + MV_VI)
    fitted = MV_X @ b
    variance = lam * MV_VI + (1 - lam) ** 2 * numpy.einsum('ij,jk,ik->i', MV_X, vb, MV_X)
    assert result.pred == pytest.approx(fitted + lam * (MV_YI - fitted), abs=1e-10)
    assert result.se == pytest.approx(numpy.sqrt(variance), abs=1e-10)


def test_mv_deletion_refits_the_components_and_refuses_a_rank_deficient_deletion() -> None:
    k = len(MV_YI)
    term = sigma2_term(list(range(k)))
    args = (MV_YI, numpy.diag(MV_VI), MV_X, [term], 'REML', [numpy.array([MV_TAU2])], [numpy.zeros(0)])
    refit = correctness.formulas.mv_deletion(*args, [numpy.array([0])], True)
    fixed = correctness.formulas.mv_deletion(*args, [numpy.array([0])], False)
    assert refit.converged == (True,)
    assert refit.cook[0] != pytest.approx(fixed.cook[0], abs=1e-6)
    only = numpy.column_stack([numpy.ones(k), numpy.array([0.0] * 6 + [1.0])])
    with pytest.raises(ValueError, match='rank deficient'):
        correctness.formulas.mv_deletion(MV_YI, numpy.diag(MV_VI), only, [term], 'REML', [numpy.array([MV_TAU2])], [numpy.zeros(0)], [numpy.array([6])], False)


def test_mv_blups_match_the_marginal_form_and_refuse_a_singular_g() -> None:
    k = len(MV_YI)
    terms = [sigma2_term([0, 0, 1, 1, 2, 2, 2]), sigma2_term(list(range(k)))]
    sigma = [numpy.array([0.07]), numpy.array([0.03])]
    cor = [numpy.zeros(0), numpy.zeros(0)]
    V = numpy.diag(MV_VI)
    result = correctness.formulas.mv_blups(MV_YI, V, MV_X, terms, sigma, cor)
    M = V + sum(correctness.formulas.term_matrix(t, v, c) for t, v, c in zip(terms, sigma, cor, strict=True))
    weight = numpy.linalg.inv(M)
    vb = numpy.linalg.inv(MV_X.T @ weight @ MV_X)
    b = vb @ MV_X.T @ weight @ MV_YI
    p_matrix = weight - weight @ MV_X @ vb @ MV_X.T @ weight
    assert result.beta == pytest.approx(b, abs=1e-10)
    for term, value, blups, pev in zip(terms, sigma, result.blups, result.pev, strict=True):
        z = (term.outer[:, None] == numpy.unique(term.outer)[None, :]).astype(numpy.float64)
        g = float(value[0]) * numpy.eye(z.shape[1])
        assert blups == pytest.approx(g @ z.T @ weight @ (MV_YI - MV_X @ b), abs=1e-10)
        assert pev == pytest.approx(numpy.diag(g - g @ z.T @ p_matrix @ z @ g), abs=1e-10)
    with pytest.raises(ValueError, match=r'term 1 \(SIGMA2\)'):
        correctness.formulas.mv_blups(MV_YI, V, MV_X, terms, [sigma[0], numpy.array([0.0])], cor)
