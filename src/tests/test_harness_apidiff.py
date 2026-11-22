"""The API step of the audit harness (``audit.apidiff``): R's formals against the Python callable for the same fit class."""

import functools
import typing

import pytest

import audit.apidiff
import audit.cases
import audit.domains
import audit.pycall
import audit.pyrun
import audit.universe
import metafopy

Entry = dict[str, typing.Any]


def fits() -> audit.pycall.Env:
    """Small fits of the bundled BCG data: its effect sizes, and a random-effects model of them."""
    wanted = {'bcg_na', 'es_bcg', 'uni_reml'}
    env, errors = audit.pyrun.build_env([f for f in audit.cases.load_fixtures() if f['name'] in wanted])
    assert errors == {}
    return env


def domains(alias: dict[str, str] | None = None) -> audit.domains.Domains:
    return audit.domains.Domains(args={}, skip={}, per_function={}, args_alias=alias or {}, graphics=[], options={}, options_skip={}, companions={})


def formals(*specs: tuple[str, str | None], dots: list[str] | None = None) -> Entry:
    return {'class': 'default', 'formals': [{'name': n, 'default': d} for n, d in specs], 'dots': dots or []}


def sample(a: int, b: str = 'x', *args: int, c: float = 1.0, **kwargs: int) -> None:
    """A callable with every parameter kind."""


class Opaque:
    """A callable whose signature cannot be read."""

    __signature__ = 'not a signature'

    def __call__(self) -> None: ...


def test_parameters_lists_names_without_the_variadic_ones() -> None:
    """parameters: ``*args`` and ``**kwargs`` are not names; ``**kwargs`` is reported."""
    assert audit.apidiff.parameters(sample, False) == ({'a', 'b', 'c'}, True)


def test_parameters_drops_the_object_parameter() -> None:
    """parameters, ``drop_first``: the first parameter, the object, is not an argument of the R function."""
    assert audit.apidiff.parameters(sample, True) == ({'b', 'c'}, True)


def test_parameters_of_a_function_without_kwargs() -> None:
    """parameters: no ``**kwargs`` gives False, and ``drop_first`` of no parameters drops nothing."""

    def plain(x: int, y: int) -> None: ...

    def none() -> None: ...

    assert audit.apidiff.parameters(plain, False) == ({'x', 'y'}, False)
    assert audit.apidiff.parameters(none, True) == (set(), False)


def test_parameters_of_a_callable_without_a_signature_accepts_anything() -> None:
    """parameters, the unreadable-signature branch: no names and ``**kwargs`` assumed."""
    assert audit.apidiff.parameters(Opaque(), False) == (set(), True)


def test_spellings_are_the_plain_name_and_the_underscored_one() -> None:
    """spellings_for, a generic with no class stem."""
    assert audit.apidiff.spellings_for('ci.lb', None, '') == ['ci_lb', 'ci_lb_']


def test_spellings_add_the_class_for_an_s3_method() -> None:
    """spellings_for, the S3 branch: ``summary.escalc`` may be ``summary_escalc``; only the part of the class before the first dot counts."""
    assert audit.apidiff.spellings_for('summary', None, 'escalc') == ['summary', 'summary_', 'summary_escalc']
    assert audit.apidiff.spellings_for('coef', None, 'summary.rma')[-1] == 'coef_summary'


@pytest.mark.parametrize('cls', ['default', 'rma', 'rma.uni', ''])
def test_spellings_ignore_the_default_and_rma_classes(cls: str) -> None:
    """spellings_for: the ``default`` and ``rma`` stems and an empty class add no spelling."""
    assert audit.apidiff.spellings_for('coef', None, cls) == ['coef', 'coef_']


def test_spellings_of_plot_follow_the_result_type() -> None:
    """spellings_for, ``plot``: the plot function of the result's type; ``plot_rma`` for a type that has none; a profile result its own."""

    class CumulResult: ...

    class LikProfileResult: ...

    class Other: ...

    assert audit.apidiff.spellings_for('plot', CumulResult(), 'cumul.rma')[-1] == 'plot_cumul'
    assert audit.apidiff.spellings_for('plot', LikProfileResult(), 'profile.rma')[-1] == 'plot_profile'
    assert audit.apidiff.spellings_for('plot', Other(), 'x')[-1] == 'plot_rma'


def test_spellings_of_plot_look_at_the_first_of_a_list() -> None:
    """spellings_for, ``plot`` of a list: the type of its first element decides; an empty list is the list itself."""

    class CumulResult: ...

    assert audit.apidiff.spellings_for('plot', [CumulResult()], 'list')[-1] == 'plot_cumul'
    assert audit.apidiff.spellings_for('plot', [], 'list')[-1] == 'plot_rma'


def test_callables_for_a_function_without_a_model() -> None:
    """callables_for, no object: the package function, not taking an object."""
    assert (metafopy.escalc, False) in audit.apidiff.callables_for('escalc', None)


def test_callables_for_an_unknown_name_is_empty() -> None:
    """callables_for: nothing in the object or the package handles the name."""
    assert audit.apidiff.callables_for('no.such.function', None) == []
    assert audit.apidiff.callables_for('no.such.function', fits()['uni_reml']) == []


def test_callables_for_a_fit_find_its_method() -> None:
    """callables_for, an object: its own bound method, which does not take the object."""
    fit = fits()['uni_reml']
    assert (fit.confint, False) in audit.apidiff.callables_for('confint', fit)


def test_callables_for_a_generic_dispatches_on_the_object_type() -> None:
    """callables_for, a singledispatch generic: the implementation for the object's type, taking the object."""
    fit = fits()['uni_reml']
    found = audit.apidiff.callables_for('confint', fit)
    assert (metafopy.confint.dispatch(type(fit)), True) in found


def test_callables_for_a_plain_function_with_an_object_takes_the_object() -> None:
    """callables_for, a package function without dispatch: it is called with the object first."""
    found = audit.apidiff.callables_for('summary', fits()['es_bcg'], 'escalc')
    assert (metafopy.summary_escalc, True) in found


def reads_level(**kwargs: int) -> int:
    return kwargs['level']


def ignores_level(**kwargs: int) -> int:
    return len(kwargs)


def test_source_mentions_finds_the_name_as_a_word() -> None:
    """source_mentions: a callable whose source names the argument reads it; a longer or shorter word does not count."""
    assert audit.apidiff.source_mentions([reads_level], 'level')
    assert not audit.apidiff.source_mentions([ignores_level], 'level')
    assert not audit.apidiff.source_mentions([reads_level], 'lev')
    assert audit.apidiff.source_mentions([ignores_level, reads_level], 'level')


def test_source_mentions_reads_through_a_partial() -> None:
    """source_mentions: the source of a ``functools.partial`` is that of the function it wraps."""
    assert audit.apidiff.source_mentions([functools.partial(reads_level)], 'level')
    assert not audit.apidiff.source_mentions([functools.partial(ignores_level)], 'level')


def test_source_mentions_assumes_yes_where_there_is_no_source() -> None:
    """source_mentions, no source: a builtin has none, so the argument is assumed read and nothing is reported."""
    assert audit.apidiff.source_mentions([len], 'level')


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (None, None),
        ('TRUE', True),
        (' FALSE ', False),
        ('NULL', None),
        ('"REML"', 'REML'),
        ('""', ''),
        ('95', 95.0),
        ('1e-4', 1e-4),
        ('c(1, 2)', ('r-expression', 'c(1, 2)')),
        ('"', ('r-expression', '"')),
    ],
)
def test_comparable_default(text: str | None, expected: object) -> None:
    """comparable_default: R's literals become Python values; anything else an expression marker that is never compared."""
    assert audit.apidiff.comparable_default(text) == expected


def test_python_defaults_lists_only_parameters_with_a_default() -> None:
    """python_defaults, the signature branch."""
    assert audit.apidiff.python_defaults(sample) == {'b': 'x', 'c': 1.0}


def test_python_defaults_reads_through_a_partial() -> None:
    """python_defaults: a ``functools.partial`` is read as the function it wraps."""
    assert audit.apidiff.python_defaults(functools.partial(sample)) == {'b': 'x', 'c': 1.0}


def test_python_defaults_of_a_callable_without_a_signature_is_empty() -> None:
    """python_defaults, the unreadable-signature branch."""
    assert audit.apidiff.python_defaults(Opaque()) == {}


def test_compare_function_reports_a_missing_function() -> None:
    """compare_function: no Python callable for R's function is a ``missing-function`` finding."""
    assert audit.apidiff.compare_function('api/f', formals(), None, 'no.such.function', {}, False) == [
        ('api/f', 'missing-function', '', 'no Python callable for no.such.function')
    ]


def test_compare_function_reports_a_missing_argument() -> None:
    """compare_function, no ``**kwargs``: an R formal with no Python parameter is ``missing-arg``."""
    findings = audit.apidiff.compare_function('api/rma', formals(('yi', None), ('nonexistent.arg', None)), None, 'rma', {}, False)
    assert [(f[0], f[1], f[2]) for f in findings if f[1] == 'missing-arg'] == [('api/rma/nonexistent.arg', 'missing-arg', 'nonexistent.arg')]


def test_compare_function_accepts_the_python_spelling_and_an_alias() -> None:
    """compare_function: a dotted R name matches its underscored parameter, and an alias maps an R name to a different one."""
    assert audit.apidiff.compare_function('api/rma', formals(('measure', None), ('knha', None)), None, 'rma', {}, False) == []
    assert audit.apidiff.compare_function('api/rma', formals(('method.name', None)), None, 'rma', {'method.name': 'method'}, False) == []


def test_compare_function_skips_the_first_formal_for_a_method() -> None:
    """compare_function, ``skip_first``: the object's formal is not an argument to look for."""
    entry = formals(('x', None), ('level', None))
    assert audit.apidiff.compare_function('api/confint', entry, fits()['uni_reml'], 'confint', {}, True) == []
    assert [f[2] for f in audit.apidiff.compare_function('api/rma', formals(('nonexistent.arg', None), ('level', None)), None, 'rma', {}, True)] == []


def test_compare_function_compares_the_arguments_r_reads_from_dots() -> None:
    """compare_function: arguments R reads from ``...`` are compared like formals."""
    findings = audit.apidiff.compare_function('api/rma', formals(dots=['nonexistent.arg']), None, 'rma', {}, False)
    assert [f[2] for f in findings] == ['nonexistent.arg']


def test_compare_function_reports_a_default_that_differs() -> None:
    """compare_function: ``method = "REML"`` in Python against ``"ML"`` in R is ``default-diff``; the same default is silent."""
    (finding,) = audit.apidiff.compare_function('api/rma', formals(('method', '"ML"')), None, 'rma', {}, False)
    assert finding[1] == 'default-diff'
    assert finding[3] == 'R default "ML", Python \'REML\''
    assert audit.apidiff.compare_function('api/rma', formals(('method', '"REML"')), None, 'rma', {}, False) == []


def test_compare_function_compares_numbers_and_booleans() -> None:
    """compare_function: R's ``95`` equals Python's ``95``, ``90`` does not; ``FALSE`` differs from Python's ``True``."""
    assert audit.apidiff.compare_function('api/rma', formals(('level', '95')), None, 'rma', {}, False) == []
    (finding,) = audit.apidiff.compare_function('api/rma', formals(('level', '90')), None, 'rma', {}, False)
    assert finding[1] == 'default-diff'
    (finding,) = audit.apidiff.compare_function('api/rma', formals(('intercept', 'FALSE')), None, 'rma', {}, False)
    assert finding[1] == 'default-diff'


def test_compare_function_does_not_compare_an_r_expression_default() -> None:
    """compare_function: a default R writes as an expression is never a difference."""
    assert audit.apidiff.compare_function('api/rma', formals(('method', 'c("a", "b")')), None, 'rma', {}, False) == []


def test_compare_function_with_kwargs_asks_whether_the_source_reads_the_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """compare_function, ``**kwargs``: an argument only ``**kwargs`` could take is missing unless a callable's source names it."""
    monkeypatch.setattr(audit.apidiff, 'callables_for', lambda name, obj, cls='': [(ignores_level, False)])
    (finding,) = audit.apidiff.compare_function('api/f', formals(('level', None)), None, 'f', {}, False)
    assert finding == ('api/f/level', 'missing-arg', 'level', 'f() takes **kwargs that never read level')
    monkeypatch.setattr(audit.apidiff, 'callables_for', lambda name, obj, cls='': [(reads_level, False)])
    assert audit.apidiff.compare_function('api/f', formals(('level', None)), None, 'f', {}, False) == []


def universe(methods: list[Entry], functions: dict[str, Entry] | None = None) -> audit.universe.Universe:
    return audit.universe.Universe(
        {'fixtures': {'uni_reml': {'class': ['rma.uni', 'rma']}, 'noclass': {'class': []}}, 'functions': functions or {}, 'methods': methods}
    )


def method(generic: str, *specs: tuple[str, str | None], cls: str = 'rma') -> Entry:
    return {'generic': generic, 'class': cls, 'formals': [{'name': n, 'default': d} for n, d in specs], 'chkclass': []}


def test_run_compares_each_generic_of_each_fixture() -> None:
    """run: for a fit, the method R dispatches to is compared with the Python callable, skipping the object's formal."""
    u = universe([method('confint', ('x', None), ('nonexistent.arg', None))])
    env = {'uni_reml': fits()['uni_reml'], 'noclass': None}
    findings = audit.apidiff.run(u, env, domains())
    assert [(f[0], f[1], f[2]) for f in findings] == [('api/confint/uni_reml/nonexistent.arg', 'missing-arg', 'nonexistent.arg')]


def test_run_skips_print_native_derived_and_refused_generics() -> None:
    """run: ``print``, a generic the harness runs natively, an ``rma`` method of a derived fixture and a generic R's class checks refuse are not compared."""
    bad = ('nonexistent.arg', None)
    refused = method('coef', ('x', None), bad)
    refused['chkclass'] = [{'must': ['rma.mv']}]
    u = universe([method('print', ('x', None), bad), method('head', ('x', None), bad), method('anova', ('x', None), bad), refused])
    env = {'uni_reml': fits()['uni_reml']}
    assert audit.apidiff.run(u, env, domains(), derived=frozenset({'uni_reml'})) == []
    # the same generics, with nothing skipped for the fixture, still leave out print, head and the refused coef
    only_anova = audit.apidiff.run(u, env, domains())
    assert {f[0] for f in only_anova} == {'api/anova/uni_reml/nonexistent.arg'}


def test_run_skips_a_generic_with_no_method_for_the_classes() -> None:
    """run: a generic whose methods are of other classes has no method for the fit, so nothing is compared."""
    u = universe([method('confint', ('x', None), ('nonexistent.arg', None), cls='rma.mv')])
    assert audit.apidiff.run(u, {'uni_reml': fits()['uni_reml']}, domains()) == []


def test_run_functions_compares_the_functions_that_take_no_model() -> None:
    """run_functions: each named R function against the Python function of the same name, with the aliases of the domains."""
    u = universe([], {'rma': formals(('yi', None), ('nonexistent.arg', None)), 'escalc': formals(('measure', None))})
    findings = audit.apidiff.run_functions(u, ['rma', 'escalc'], domains())
    assert [f[0] for f in findings] == ['api/rma/nonexistent.arg']
