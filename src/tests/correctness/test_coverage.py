"""The scope gate of the correctness suite and the checks of the harness itself.

The gate names every item of the core that has no ``published`` or ``derived`` case.
"""

import ast
import dataclasses
import pathlib
import types
import typing

import numpy
import pytest

import correctness.cases
import correctness.glmm_formulas
import correctness.test_published
import metafopy
import metafopy.core.clubsandwich
import metafopy.core.glm
import metafopy.datasets
import metafopy.effects.measures
import metafopy.errors
import metafopy.methods.confint
import metafopy.methods.robust
import metafopy.models.glmm_fit
import metafopy.models.mv_structs
import metafopy.models.rma_glmm
import metafopy.models.rma_mv
import metafopy.models.rma_uni
import metafopy.models.tau2

HETEROGENEITY = ('H2', 'I2', 'QE', 'QM')
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
    items += glmm_items()
    return items


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
        items += [f'rma_glmm.fit.{measure}.{model}.{method}' for model in models for method in ('FE', 'ML')]
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
                bi='bi',
                ci='ci',
                di='di',
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


@pytest.mark.xfail(strict=True, reason='rma_glmm coverage incomplete; see agents/tasks/correctness-glmm/tasks-1.md')
def test_every_core_item_has_a_case() -> None:
    missing = missing_items(correctness.cases.load_cases(), correctness.cases.load_exemptions())
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
        (VALID + VALID, r"inline\.toml: duplicate case id 'unit\.md'"),
        ('[[case]\nid = 1', r'inline\.toml: invalid TOML'),
        ('[mystery]\nx = 1', r'inline\.toml: unknown top-level key'),
    ],
)
def test_an_invalid_case_file_raises_naming_file_and_case(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        correctness.cases.parse_cases(text, 'inline.toml')


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


def test_build_v_refuses_a_block_of_the_wrong_size_or_one_not_symmetric() -> None:
    case = one_case(VBLOCKS_CASE)
    assert case.setup is not None
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    where = 'inline.toml: case unit.vblocks'
    with pytest.raises(ValueError, match=r'inline\.toml: case unit\.vblocks: cluster 2\.0 has 1 rows but vblocks names 2 columns'):
        correctness.cases.build_v(frame.iloc[:3], case.setup, where)
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
    built = correctness.cases.build_tree_r('moura2021')
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
