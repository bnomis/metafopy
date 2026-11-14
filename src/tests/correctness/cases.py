"""Loader and comparison helpers for the correctness cases in ``cases/*.toml``.

A case names a metafopy function, the data and arguments to call it with, and the values it must return. The values come from outside metafopy and
metafor: a number printed in the defining publication (``published``), a formula coded from the publication in ``formulas.py`` (``derived``), an identity
theory guarantees (``property``), a seeded Monte Carlo check (``simulation``) or another implementation (``other``). See
``agents/specs/correctness-tests/spec.md``.

An ``expect`` key names a result attribute (``tau2``), a column of an ``escalc`` frame (``yi``) or one element of either (``yi[0]``, 0-based). The value
of a ``published`` key is the string as printed, so that the tolerance follows from its last digit. The value of a ``derived`` key is a float or the
name of a function in ``formulas.py`` (``formulas.rd_yi``), which is called with the case's ``args`` it declares, a column name replaced by the column.
"""

import collections.abc
import dataclasses
import datetime
import decimal
import importlib
import inspect
import pathlib
import re
import tomllib
import typing

import numpy
import pandas
import pytest

import correctness.formulas
import metafopy
import metafopy.datasets

type Kind = typing.Literal['published', 'derived', 'property', 'simulation', 'other']
type Status = typing.Literal['pass', 'mismatch']

CASES_DIR = pathlib.Path(__file__).parent / 'cases'
KINDS: tuple[Kind, ...] = typing.get_args(Kind.__value__)
STATUSES: tuple[Status, ...] = typing.get_args(Status.__value__)
GATE_KINDS: tuple[Kind, ...] = ('derived', 'published')

ID_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
ACCESSOR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?')
FUNCTION_PATTERN = re.compile(r'(?:formulas\.)?[A-Za-z_][A-Za-z0-9_]*')
CHECK_PATTERN = re.compile(r'test_[a-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*')
FORMULA_PREFIX = 'formulas.'

# auditme: accept[non-alphabetical-literal] order follows the spec's description of a case
CASE_KEYS = frozenset(
    {'args', 'check', 'covers', 'data', 'dataset', 'expect', 'function', 'id', 'kind', 'printed', 'reason', 'setup', 'source', 'status', 'tolerance'}
)
SOURCE_KEYS = frozenset({'citation', 'doi', 'locator', 'read', 'url'})
TOP_KEYS = frozenset({'case', 'exempt', 'vtypes'})
DERIVED_TOLERANCE = 1e-10
PROPERTY_TOLERANCE = 1e-8
OTHER_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
class Source:
    """Where a case's expected values come from: citation, DOI or URL, locator in the source, date read."""

    citation: str
    locator: str
    read: datetime.date
    doi: str | None = None
    url: str | None = None


@dataclasses.dataclass(frozen=True)
class Case:
    """One correctness case from a ``cases/*.toml`` file.

    ``covers`` names the coverage items (see ``test_coverage.py``) the case checks. ``data`` is an inline dataset, used when ``dataset`` is not given.
    ``file`` is the name of the case file, for messages. ``printed`` holds values a source prints that no attribute of the result holds (a value on
    another scale, an interval of ``predict`` or ``confint``), as printed; the case's ``check`` function compares them with ``check_printed``.
    """

    id: str
    kind: Kind
    function: str
    args: dict[str, typing.Any]
    expect: dict[str, str | float]
    status: Status
    source: Source | None
    dataset: str | None = None
    data: dict[str, list[float]] | None = None
    setup: dict[str, typing.Any] | None = None
    reason: str = ''
    tolerance: float | None = None
    check: str | None = None
    covers: tuple[str, ...] = ()
    file: str = ''
    printed: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Exemption:
    """A coverage item that needs no ``published`` or ``derived`` case, with the reason."""

    item: str
    reason: str
    file: str


@dataclasses.dataclass(frozen=True)
class MeasureVtypes:
    """A measure of ``escalc`` and the ``vtype`` values it accepts."""

    measure: str
    vtype: tuple[str, ...]
    file: str


def published_tolerance(expected: str) -> float:
    """Half a unit in the last printed digit of ``expected`` (``'0.0193'`` -> ``5e-05``; ``'684'`` -> ``0.5``)."""
    try:
        number = decimal.Decimal(expected.strip())
    except decimal.InvalidOperation as err:
        raise ValueError(f'{expected!r} is not a printed number') from err
    exponent = number.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f'{expected!r} is not a finite printed number')
    return float(decimal.Decimal(5).scaleb(exponent - 1))


def fail(where: str, message: str) -> typing.NoReturn:
    raise ValueError(f'{where}: {message}')


def parse_source(raw: object, where: str) -> Source:
    if not isinstance(raw, dict):
        fail(where, 'source must be a table')
    unknown = set(raw) - SOURCE_KEYS
    if unknown:
        fail(where, f'unknown source key(s) {sorted(unknown)}')
    for key in ('citation', 'locator', 'read'):
        if key not in raw:
            fail(where, f'source has no {key!r}')
    read = raw['read']
    if isinstance(read, str):
        try:
            read = datetime.date.fromisoformat(read)
        except ValueError:
            fail(where, f'source.read {read!r} is not a date')
    if not isinstance(read, datetime.date) or isinstance(read, datetime.datetime):
        fail(where, 'source.read must be a date')
    for key in ('citation', 'locator'):
        if not isinstance(raw[key], str) or not raw[key].strip():
            fail(where, f'source.{key} must be a non-empty string')
    return Source(citation=raw['citation'], locator=raw['locator'], read=read, doi=raw.get('doi'), url=raw.get('url'))


def parse_expect(kind: str, raw: object, where: str) -> dict[str, str | float]:
    if not isinstance(raw, dict):
        fail(where, 'expect must be a table')
    expect: dict[str, str | float] = {}
    for key, value in raw.items():
        if not ACCESSOR.fullmatch(key):
            fail(where, f'expect key {key!r} is not an attribute name with an optional [index]')
        if kind == 'published':
            if not isinstance(value, str):
                fail(where, f'published value of {key!r} must be a string, as printed in the source')
            published_tolerance(value)
        elif kind == 'derived' and isinstance(value, str):
            if not value.startswith(FORMULA_PREFIX) or not FUNCTION_PATTERN.fullmatch(value):
                fail(where, f'derived value of {key!r} must be a float or the name of a function "formulas.<name>"')
        elif isinstance(value, bool) or not isinstance(value, int | float):
            fail(where, f'value of {key!r} must be a float')
        expect[key] = value
    return expect


def parse_printed(raw: object, kind: str, check: str | None, where: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        fail(where, 'printed must be a table')
    if raw and (kind != 'published' or check is None):
        fail(where, 'printed is allowed only with kind "published" and a check')
    printed: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            fail(where, f'printed value of {key!r} must be a string, as printed in the source')
        try:
            published_tolerance(value)
        except ValueError:
            fail(where, f'printed value of {key!r} ({value!r}) is not a printed number')
        printed[key] = value
    return printed


def parse_case(raw: object, name: str, index: int) -> Case:
    if not isinstance(raw, dict):
        fail(name, f'case {index} must be a table')
    ident = raw.get('id')
    if not isinstance(ident, str) or not ID_PATTERN.fullmatch(ident):
        fail(name, f'case {index} has a missing or invalid id {ident!r} (letters, digits, ".", "-" and "_" only)')
    where = f'{name}: case {ident}'
    unknown = set(raw) - CASE_KEYS
    if unknown:
        fail(name, f'case {ident}: unknown key(s) {sorted(unknown)}')
    for key in ('kind', 'function'):
        if key not in raw:
            fail(name, f'case {ident}: missing {key!r}')
    kind = raw['kind']
    if kind not in KINDS:
        fail(name, f'case {ident}: unknown kind {kind!r}')
    function = raw['function']
    if not isinstance(function, str) or not FUNCTION_PATTERN.fullmatch(function):
        fail(name, f'case {ident}: function {function!r} must be a metafopy function name or "formulas.<name>"')
    check = raw.get('check')
    if check is not None and (not isinstance(check, str) or not CHECK_PATTERN.fullmatch(check)):
        fail(name, f'case {ident}: check {check!r} must be "<test module>.<function>" in src/tests/correctness')
    if 'expect' not in raw and check is None:
        fail(name, f'case {ident}: missing "expect" (or a "check")')
    expect = parse_expect(kind, raw.get('expect', {}), where)
    printed = parse_printed(raw.get('printed', {}), kind, check, where)
    if kind == 'published' and not expect and not printed:
        fail(name, f'case {ident}: a published case needs expect or printed values')
    status = raw.get('status', 'pass')
    if status not in STATUSES:
        fail(name, f'case {ident}: status {status!r} must be "pass" or "mismatch"')
    reason = raw.get('reason', '')
    if status == 'mismatch' and not (isinstance(reason, str) and reason.strip()):
        fail(name, f'case {ident}: a mismatch case needs a reason naming both values')
    source = parse_source(raw['source'], where) if 'source' in raw else None
    if kind in ('published', 'other') and source is None:
        fail(name, f'case {ident}: a {kind} case needs a source')
    tolerance = raw.get('tolerance')
    if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, int | float) or not tolerance > 0):
        fail(name, f'case {ident}: tolerance must be a positive number')
    if kind == 'simulation' and tolerance is None:
        fail(name, f'case {ident}: a simulation case states its tolerance')
    setup = raw.get('setup')
    if setup is not None and (not isinstance(setup, dict) or set(setup) != {'escalc'} or not isinstance(setup['escalc'], dict)):
        fail(name, f'case {ident}: setup must be {{ escalc = {{ ... }} }}')
    data = raw.get('data')
    if data is not None and not isinstance(data, dict):
        fail(name, f'case {ident}: data must be a table of columns')
    if data is not None and 'dataset' in raw:
        fail(name, f'case {ident}: give a dataset or inline data, not both')
    covers = raw.get('covers', [])
    if not isinstance(covers, list) or not all(isinstance(c, str) for c in covers):
        fail(name, f'case {ident}: covers must be a list of coverage items')
    return Case(
        id=ident,
        kind=kind,
        function=function,
        args=dict(raw.get('args', {})),
        expect=expect,
        status=status,
        source=source,
        dataset=raw.get('dataset'),
        data=data,
        setup=setup,
        reason=reason,
        tolerance=None if tolerance is None else float(tolerance),
        check=check,
        covers=tuple(covers),
        file=name,
        printed=printed,
    )


def parse_document(text: str, name: str) -> dict[str, typing.Any]:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        # auditme: accept[except-interpolation] the decoder's message holds the line and column, the only locator inside the file
        fail(name, f'invalid TOML: {err}')
    unknown = set(doc) - TOP_KEYS
    if unknown:
        fail(name, f'unknown top-level key(s) {sorted(unknown)}')
    return doc


def parse_cases(text: str, name: str) -> list[Case]:
    """Parse and validate the cases of one case file given as ``text``; ``name`` is the file name used in messages."""
    cases: list[Case] = []
    seen: set[str] = set()
    for index, raw in enumerate(parse_document(text, name).get('case', [])):
        case = parse_case(raw, name, index)
        if case.id in seen:
            fail(name, f'duplicate case id {case.id!r}')
        seen.add(case.id)
        cases.append(case)
    return cases


def case_files(directory: pathlib.Path | None = None) -> list[pathlib.Path]:
    return sorted((directory or CASES_DIR).glob('*.toml'))


def load_cases(kind: str | None = None, *, directory: pathlib.Path | None = None) -> list[Case]:
    """Read every ``cases/*.toml`` file, validate each case, and return them (filtered by ``kind``)."""
    cases: list[Case] = []
    first_file: dict[str, str] = {}
    for path in case_files(directory):
        for case in parse_cases(path.read_text(encoding='utf-8'), path.name):
            if case.id in first_file:
                fail(path.name, f'duplicate case id {case.id!r}, also in {first_file[case.id]}')
            first_file[case.id] = path.name
            cases.append(case)
    return [c for c in cases if kind is None or c.kind == kind]


def load_exemptions(directory: pathlib.Path | None = None) -> list[Exemption]:
    """The ``[[exempt]]`` entries of every case file: a coverage item and the reason it needs no case."""
    out: list[Exemption] = []
    for path in case_files(directory):
        for raw in parse_document(path.read_text(encoding='utf-8'), path.name).get('exempt', []):
            if not isinstance(raw, dict) or set(raw) != {'item', 'reason'} or not all(isinstance(v, str) and v.strip() for v in raw.values()):
                fail(path.name, f'an [[exempt]] entry needs a non-empty "item" and "reason", got {raw!r}')
            out.append(Exemption(item=raw['item'], reason=raw['reason'], file=path.name))
    return out


def load_vtypes(directory: pathlib.Path | None = None) -> list[MeasureVtypes]:
    """The ``[[vtypes]]`` entries of every case file: an ``escalc`` measure and the ``vtype`` values it accepts."""
    out: list[MeasureVtypes] = []
    for path in case_files(directory):
        for raw in parse_document(path.read_text(encoding='utf-8'), path.name).get('vtypes', []):
            vtype = raw.get('vtype') if isinstance(raw, dict) else None
            if not isinstance(raw, dict) or set(raw) != {'measure', 'vtype'} or not isinstance(raw['measure'], str):
                fail(path.name, f'a [[vtypes]] entry needs a "measure" and a list "vtype", got {raw!r}')
            if not isinstance(vtype, list) or not vtype or not all(isinstance(v, str) for v in vtype):
                fail(path.name, f'[[vtypes]] {raw["measure"]}: "vtype" must be a non-empty list of strings')
            out.append(MeasureVtypes(measure=raw['measure'], vtype=tuple(vtype), file=path.name))
    return out


def case_frame(case: Case) -> pandas.DataFrame | None:
    """The case's data: the bundled dataset or the inline columns, with the ``escalc`` of ``setup`` applied."""
    if case.dataset is not None:
        frame = metafopy.datasets.load(case.dataset)
    elif case.data is not None:
        frame = pandas.DataFrame(case.data)
    else:
        return None
    if case.setup is not None:
        frame = metafopy.escalc(data=frame, **case.setup['escalc'])
    return frame


def resolve_formula(name: str, where: str) -> collections.abc.Callable[..., typing.Any]:
    func = getattr(correctness.formulas, name.removeprefix(FORMULA_PREFIX), None)
    if not callable(func):
        raise LookupError(f'{where}: formulas.py has no function {name.removeprefix(FORMULA_PREFIX)!r}')
    return typing.cast(collections.abc.Callable[..., typing.Any], func)


def formula_value(case: Case, name: str, frame: pandas.DataFrame | None) -> typing.Any:
    """Call a formula with the ``args`` of the case that it declares, a column name replaced by the column."""
    func = resolve_formula(name, case.id)
    kwargs: dict[str, typing.Any] = {}
    for param in inspect.signature(func).parameters:
        if param == 'data' and frame is not None:
            kwargs[param] = frame
            continue
        if param not in case.args:
            continue
        value = case.args[param]
        if isinstance(value, str) and frame is not None and value in frame.columns:
            value = frame[value].to_numpy(dtype=numpy.float64)
        kwargs[param] = value
    return func(**kwargs)


def run_case(case: Case) -> typing.Any:
    """Load the dataset, apply ``setup`` (an ``escalc`` call), and call ``metafopy.<function>(**args)`` (``formulas.<name>`` for a formula)."""
    frame = case_frame(case)
    if case.function.startswith(FORMULA_PREFIX):
        return formula_value(case, case.function, frame)
    func = getattr(metafopy, case.function, None)
    if not callable(func):
        raise LookupError(f'{case.id}: metafopy has no function {case.function!r}')
    kwargs = dict(case.args)
    if frame is not None:
        kwargs['data'] = frame
    return func(**kwargs)


def fetch(result: typing.Any, key: str, where: str) -> typing.Any:
    """The value ``key`` names in ``result``: an attribute, a frame column or a mapping entry, with an optional 0-based ``[i]``."""
    match = ACCESSOR.fullmatch(key)
    if match is None:
        raise LookupError(f'{where}: {key!r} is not an attribute name with an optional [index]')
    name, index = match.groups()
    if (isinstance(result, pandas.DataFrame) and name in result.columns) or (isinstance(result, collections.abc.Mapping) and name in result):
        value = result[name]
    elif hasattr(result, name):
        value = getattr(result, name)
    else:
        raise LookupError(f'{where}: the result has no attribute {name!r}')
    array = numpy.asarray(value, dtype=numpy.float64)
    return float(array.reshape(-1)[int(index)]) if index is not None else array


def expected_value(case: Case, key: str, frame: pandas.DataFrame | None) -> tuple[typing.Any, float, str]:
    """The reference value for ``key``, the allowed difference, and where the value comes from."""
    raw = case.expect[key]
    if isinstance(raw, str) and case.kind == 'published':
        assert case.source is not None
        return float(raw), published_tolerance(raw), f'{case.source.citation}, {case.source.locator}, read {case.source.read}'
    if isinstance(raw, str):
        value = formula_element(formula_value(case, raw, frame), key)
        return value, case.tolerance or DERIVED_TOLERANCE, raw
    default = {'derived': DERIVED_TOLERANCE, 'other': OTHER_TOLERANCE}.get(case.kind, PROPERTY_TOLERANCE)
    origin = f'{case.source.citation}, {case.source.locator}, read {case.source.read}' if case.source else case.kind
    return float(raw), case.tolerance or default, origin


def formula_element(value: typing.Any, key: str) -> typing.Any:
    """The element of a formula's value that ``key`` selects (``yi[0]``), or all of it."""
    index = ACCESSOR.fullmatch(key)
    array = numpy.asarray(value, dtype=numpy.float64)
    if index is not None and index.group(2) is not None:
        return float(array.reshape(-1)[int(index.group(2))])
    if array.ndim == 0:
        return float(array)
    return array


def check_case(case: Case, result: typing.Any) -> None:
    """Compare each ``expect`` key with the result at the kind's tolerance; raise AssertionError naming case, key, both values and the source."""
    frame = case_frame(case) if any(isinstance(v, str) and v.startswith(FORMULA_PREFIX) for v in case.expect.values()) else None
    for key in case.expect:
        actual = fetch(result, key, case.id)
        expected, tolerance, origin = expected_value(case, key, frame)
        actual_array, expected_array = numpy.atleast_1d(actual), numpy.atleast_1d(expected)
        try:
            actual_array, expected_array = numpy.broadcast_arrays(actual_array, expected_array)
        except ValueError:
            raise AssertionError(
                f'{case.id}: {key}: metafopy gives shape {actual_array.shape}, the reference shape {expected_array.shape} ({origin})'
            ) from None
        allowed = tolerance if case.kind == 'published' else tolerance * numpy.maximum(1.0, numpy.abs(expected_array))
        both_nan = numpy.isnan(actual_array) & numpy.isnan(expected_array)
        close = both_nan | (numpy.abs(actual_array - expected_array) <= allowed)
        if not close.all():
            raise AssertionError(f'{case.id}: {key}: metafopy gives {actual!r}, the reference gives {expected!r} (tolerance {tolerance:g}; source: {origin})')
    if case.check is not None:
        module_name, function_name = case.check.split('.')
        func = getattr(importlib.import_module(f'correctness.{module_name}'), function_name, None)
        if not callable(func):
            raise LookupError(f'{case.id}: {module_name} has no check function {function_name!r}')
        func(case, result)


def check_printed(case: Case, name: str, actual: float) -> None:
    """Compare ``actual`` with the printed value ``name`` of the case at the printed tolerance; raise AssertionError naming case, name, both and the source."""
    if name not in case.printed:
        raise LookupError(f'{case.id}: no printed value {name!r}')
    assert case.source is not None
    raw = case.printed[name]
    expected, tolerance = float(raw), published_tolerance(raw)
    if not abs(actual - expected) <= tolerance:
        origin = f'{case.source.citation}, {case.source.locator}, read {case.source.read}'
        raise AssertionError(f'{case.id}: {name}: metafopy gives {actual!r}, the reference gives {expected!r} (tolerance {tolerance:g}; source: {origin})')


def case_param(case: Case) -> typing.Any:
    """The case as a pytest param (id = case id); a ``mismatch`` case is marked ``xfail(strict=True, reason=...)``."""
    marks = [pytest.mark.xfail(strict=True, reason=case.reason)] if case.status == 'mismatch' else []
    return pytest.param(case, id=case.id, marks=marks)


def case_params(kind: str) -> list[typing.Any]:
    """Cases of one kind as pytest params, id = case id; a ``mismatch`` case is marked ``xfail(strict=True, reason=...)``."""
    return [case_param(c) for c in load_cases(kind)]
