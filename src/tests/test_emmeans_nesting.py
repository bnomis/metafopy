"""Nested factors in emmprep(), emmeans() and contrast() against R's emmeans package (src/tests/r/generate_emmeans.R)."""

import collections.abc

import emmcheck
import numpy
import pandas
import pytest

import metafopy
import metafopy.errors
import metafopy.methods.emm_nesting
import metafopy.methods.emm_summary

type Case = collections.abc.Callable[[metafopy.EmmGrid, str, metafopy.EmmGrid], metafopy.methods.emm_summary.EmmSummary]

WEIGHTS = ('proportional', 'outer', 'cells', 'flat')

SPECS = {
    'n1': {'grp': 'grp', 'grpby': '~ grp | lat', 'lat': 'lat', 'both': '~ lat * grp', 'one': '~ 1', 'latby': '~ lat | grp'},
    'n2': {'grp': 'grp3', 'grpby': '~ grp3 | lat', 'lat': 'lat', 'both': '~ lat * grp3', 'one': '~ 1', 'latby': '~ lat | grp3'},
    'n3': {
        'alloc': 'alloc',
        'grp': 'grp',
        'allocgrp': '~ alloc * grp',
        'allocby': '~ alloc | grp',
        'grpby': '~ grp | alloc',
        'lat': 'lat',
        'all': '~ alloc * lat * grp',
        'one': '~ 1',
    },
    'n4': {'sub': 'sub', 'subby': '~ sub | grp * lat', 'grp': 'grp', 'lat': 'lat', 'all': '~ lat * grp * sub', 'one': '~ 1'},
}


def means(g: metafopy.EmmGrid, spec: str, em: metafopy.EmmGrid) -> metafopy.methods.emm_summary.EmmSummary:
    return em.summary(infer=True)


CASES: dict[str, Case] = {
    'emm': means,
    'emm_resp': lambda g, spec, em: em.summary(infer=True, type='response'),
    'pairs': lambda g, spec, em: metafopy.pairs(em).summary(infer=True),
    'pairs_resp': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, type='response'),
    'pairs_bonf': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, adjust='bonferroni'),
    'pairs_sidak': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, adjust='sidak'),
    'pairs_none': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, adjust='none'),
    'pairs_scheffe': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, adjust='scheffe'),
    'revpairs': lambda g, spec, em: metafopy.pairs(em, reverse=True).summary(infer=True),
    'eff': lambda g, spec, em: metafopy.contrast(em, 'eff').summary(infer=True),
    'trt': lambda g, spec, em: metafopy.contrast(em, 'trt_vs_ctrl').summary(infer=True),
    'trt_ref2': lambda g, spec, em: metafopy.contrast(em, 'trt_vs_ctrl', ref=1).summary(infer=True),
    'consec': lambda g, spec, em: metafopy.contrast(em, 'consec').summary(infer=True),
    'dunnett_resp': lambda g, spec, em: metafopy.contrast(em, 'dunnett').summary(infer=True, type='response'),
    'pairs_side': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, side='<', adjust='holm'),
    'pairs_delta': lambda g, spec, em: metafopy.pairs(em).summary(infer=True, delta=0.3, adjust='none'),
    'regrid': lambda g, spec, em: metafopy.regrid(em).summary(infer=True),
    'regrid_pairs': lambda g, spec, em: metafopy.pairs(metafopy.regrid(em)).summary(infer=True),
    'pairs_by_grp': lambda g, spec, em: metafopy.pairs(em, by='grp').summary(infer=True),
    'pairs_by_lat': lambda g, spec, em: metafopy.pairs(em, by='lat').summary(infer=True),
}


def weighted(weight: str) -> Case:
    return lambda g, spec, em: metafopy.emmeans(g, spec, weights=weight).summary(infer=True)


for weight in WEIGHTS:
    CASES[f'w_{weight}'] = weighted(weight)


@pytest.fixture(scope='module')
def designs() -> dict[str, metafopy.EmmGrid]:
    dat, cells = emmcheck.nested_data()
    n1 = metafopy.rma_uni('yi', 'vi', mods='~ lat + lat:grp', data=dat)
    n2 = metafopy.rma_uni('yi', 'vi', mods='~ lat + grp3', data=dat)
    n3 = metafopy.rma_uni('yi', 'vi', mods='~ alloc + lat + lat:grp + ablat', data=dat)
    n4 = metafopy.rma_uni('yi', 'vi', mods='~ lat + lat:grp + lat:grp:sub', data=cells)
    return {
        'n1': metafopy.emmprep(n1),
        'n2': metafopy.emmprep(n2, nesting='grp3 %in% lat'),
        'n3': metafopy.emmprep(n3),
        'n4': metafopy.emmprep(n4),
    }


@pytest.mark.parametrize(
    ('prefix', 'name'),
    [(prefix, name) for prefix, specs in SPECS.items() for name in specs],
)
def test_nested_means_and_contrasts(designs: dict[str, metafopy.EmmGrid], prefix: str, name: str) -> None:
    g = designs[prefix]
    spec = SPECS[prefix][name]
    em = metafopy.emmeans(g, spec)
    failures: list[str] = []
    seen = 0
    for suffix, case in CASES.items():
        key = f'{prefix}_{name}_{suffix}'
        if key not in emmcheck.REF:
            continue
        seen += 1
        try:
            emmcheck.check(case(g, spec, em), key, emmcheck.MVT if suffix == 'consec' else emmcheck.RTOL)
        except AssertionError as err:
            failures.append(f'{key}: {str(err)[:300]}')  # auditme: accept the failure text is the report of a collected check
    assert seen > 0
    assert not failures, '\n'.join(failures)


def test_detected_nesting(designs: dict[str, metafopy.EmmGrid]) -> None:
    assert designs['n1'].nesting == {'grp': ('lat',)}
    assert designs['n1'].display == (True,) * 4
    assert designs['n2'].nesting == {'grp3': ('lat',)}
    assert designs['n2'].display == (True, True, True, True, True, False)
    assert designs['n3'].nesting == {'grp': ('lat',)}
    assert designs['n4'].nesting == {'grp': ('lat',), 'sub': ('lat', 'grp')}


def test_hidden_combinations_are_not_shown(designs: dict[str, metafopy.EmmGrid]) -> None:
    both = metafopy.emmeans(designs['n2'], '~ lat * grp3').summary(infer=True)
    assert len(both.frame) == 5
    assert not ((both.frame['lat'] == 'low') & (both.frame['grp3'] == 'c')).any()
    assert metafopy.emmeans(designs['n2'], 'lat').display is None


def test_nesting_options() -> None:
    dat, _ = emmcheck.nested_data()
    n1 = metafopy.rma_uni('yi', 'vi', mods='~ lat + lat:grp', data=dat)
    n2 = metafopy.rma_uni('yi', 'vi', mods='~ lat + grp3', data=dat)
    off = metafopy.emmprep(n1, nesting=None)
    assert off.nesting == {}
    assert off.display is None
    emmcheck.check(metafopy.emmeans(off, 'grp').summary(infer=True), 'n1_off_grp')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(off, '~ lat * grp')).summary(infer=True), 'n1_off_pairs')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(n1, at={'grp': 'a'}), 'lat').summary(infer=True), 'n1_at_grp')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(n2, nesting={'grp3': 'lat'}), 'grp3').summary(infer=True), 'n2_list_grp')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(n2), 'grp3').summary(infer=True), 'n2_off_grp')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(n2), 'lat').summary(infer=True), 'n2_off_lat')
    with pytest.raises(metafopy.errors.InputError, match='Nonexistent variables'):
        metafopy.emmprep(n2, nesting='nothing %in% lat')
    with pytest.raises(metafopy.errors.InputError, match='Cannot read the nesting'):
        metafopy.emmprep(n2, nesting='grp3 in lat')


def test_nested_weights(designs: dict[str, metafopy.EmmGrid]) -> None:
    g = designs['n2']
    assert 'n2_w_vector' not in emmcheck.REF
    with pytest.raises(metafopy.errors.InputError, match='Nonconforming number of weights -- need 2'):
        metafopy.emmeans(g, 'lat', weights=[1, 1, 1])
    with pytest.raises(metafopy.errors.InputError, match='Nonconforming number of weights -- need 2'):
        metafopy.emmeans(g, 'lat', weights=numpy.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))


def test_nested_contrast_errors(designs: dict[str, metafopy.EmmGrid]) -> None:
    n1 = designs['n1']
    latby = metafopy.emmeans(n1, '~ lat | grp')
    assert latby.by == ('lat', 'grp')
    with pytest.raises(metafopy.errors.InputError, match='no factor levels left'):
        metafopy.pairs(latby)
    with pytest.raises(metafopy.errors.InputError, match='no factor levels left'):
        metafopy.pairs(metafopy.emmeans(n1, 'grp'), by='grp')
    with pytest.raises(metafopy.errors.InputError, match='Non-character contrast methods'):
        metafopy.contrast(metafopy.emmeans(n1, 'grp'), {'a vs b': [1.0, -1.0, 0.0, 0.0]})
    with pytest.raises(metafopy.errors.InputError, match='not all in the grid'):
        metafopy.pairs(metafopy.emmeans(n1, 'grp'), by='nothing')


def test_contrasts_by_the_groups_leave_no_nesting(designs: dict[str, metafopy.EmmGrid]) -> None:
    by_lat = metafopy.pairs(metafopy.emmeans(designs['n1'], 'grp'), by='lat')
    assert by_lat.by == ('lat',)
    assert by_lat.nesting == {}
    assert by_lat.display is None


def test_term_codes() -> None:
    info = metafopy.methods.emm_nesting.term_info('lat + lat:grp', ['lat', 'grp'])
    assert info.codes == ({'lat': 1}, {'lat': 2, 'grp': 1})
    assert info.main == {'lat'}
    crossed = metafopy.methods.emm_nesting.term_info('lat * grp', ['lat', 'grp'])
    assert crossed.codes == ({'lat': 1}, {'grp': 1}, {'lat': 1, 'grp': 1})
    no_intercept = metafopy.methods.emm_nesting.term_info('0 + lat + lat:grp', ['lat', 'grp'])
    assert no_intercept.codes[0] == {'lat': 2}
    wrapped = metafopy.methods.emm_nesting.term_info('C(lat) + I(x ** 2)', ['lat', 'x'])
    assert wrapped.main == {'lat'}
    assert wrapped.codes == ({'lat': 1}, {})


def test_nesting_from_the_data() -> None:
    frame = pandas.DataFrame({'a': ['x', 'x', 'y', 'y'], 'b': ['p', 'q', 'r', 'r']})
    levels = {'a': ('x', 'y'), 'b': ('p', 'q', 'r')}
    wgt = numpy.array([1.0, 1.0, 1.0, 1.0])
    assert metafopy.methods.emm_nesting.find_nests(frame, wgt, levels, None) == {'b': ('a',)}
    assert metafopy.methods.emm_nesting.find_nests(frame, wgt, {'a': ('x', 'y')}, None) == {}


def test_parse_nest() -> None:
    parse = metafopy.methods.emm_nesting.parse_nest
    assert parse('B %in% A') == {'B': ('A',)}
    assert parse('B %in% (A * C), D %in% B') == {'B': ('A', 'C'), 'D': ('B',)}
    assert parse({'B': 'A', 'D': ['A', 'B']}) == {'B': ('A',), 'D': ('A', 'B')}
