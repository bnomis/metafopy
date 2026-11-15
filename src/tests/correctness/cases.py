"""Loader and comparison helpers for the correctness cases in ``cases/*.toml``.

A case names a metafopy function, the data and arguments to call it with, and the values it must return. The values come from outside metafopy and
metafor: a number printed in the defining publication (``published``), a formula coded from the publication in ``formulas.py`` or ``glmm_formulas.py``
(``derived``), an identity theory guarantees (``property``), a seeded Monte Carlo check (``simulation``) or another implementation (``other``). See
``agents/specs/correctness-tests/spec.md``.

An ``expect`` key names a result attribute (``tau2``), a column of an ``escalc`` frame (``yi``) or one element of either (``yi[0]``, 0-based). The value
of a ``published`` key is the string as printed, so that the tolerance follows from its last digit. The value of a ``derived`` key is a float or the
name of a function in a module of ``FORMULA_MODULES`` (``formulas.rd_yi``, ``glmm_formulas.<name>``), which is called with the case's ``args`` it
declares, a column name replaced by the column.
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
import numpy.typing
import pandas
import pytest

import correctness.formulas
import correctness.glmm_formulas
import metafopy
import metafopy.datasets

type Kind = typing.Literal['published', 'derived', 'property', 'simulation', 'other']
type Status = typing.Literal['pass', 'mismatch']
type FloatArray = numpy.typing.NDArray[numpy.float64]

FORMULA_MODULE_OBJECTS = {'formulas': correctness.formulas, 'glmm_formulas': correctness.glmm_formulas}
CASES_DIR = pathlib.Path(__file__).parent / 'cases'
KINDS: tuple[Kind, ...] = typing.get_args(Kind.__value__)
STATUSES: tuple[Status, ...] = typing.get_args(Status.__value__)
GATE_KINDS: tuple[Kind, ...] = ('derived', 'published')

ID_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
ACCESSOR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?')
FUNCTION_PATTERN = re.compile(r'(?:[a-z_]+\.)?[A-Za-z_][A-Za-z0-9_]*')
CHECK_PATTERN = re.compile(r'test_[a-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*')
# auditme: accept[non-alphabetical-literal] the order is the order the modules were added
FORMULA_MODULES = ('formulas', 'glmm_formulas')

# auditme: accept[non-alphabetical-literal] order follows the spec's description of a case
CASE_KEYS = frozenset(
    {
        'args',
        'check',
        'covers',
        'data',
        'dataset',
        'expect',
        'function',
        'id',
        'kind',
        'printed',
        'reason',
        'setup',
        'slow',
        'source',
        'status',
        'then',
        'tolerance',
    }
)
SETUP_KEYS = frozenset({'drop', 'escalc', 'rma_uni', 'subset', 'tree_r', 'vblocks', 'vconst', 'where'})
STEP_KEYS = frozenset({'args', 'function'})
VBLOCKS_KEYS = frozenset({'cluster', 'columns'})
VCONST_KEYS = frozenset({'cluster', 'rho', 'vi'})
SUBSET_KEYS = frozenset({'column', 'equals'})
TREE_R_KEYS = frozenset({'factor', 'tree'})
SYMMETRY_TOLERANCE = 1e-12
SOURCE_KEYS = frozenset({'citation', 'doi', 'locator', 'read', 'url'})
TOP_KEYS = frozenset({'case', 'derived_required', 'exempt', 'vtypes'})
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
class Step:
    """A follow-on call: ``metafopy.<function>(result, **args)`` on the result of the case's own call."""

    function: str
    args: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


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
    then: Step | None = None
    slow: bool = False


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
            if not is_formula(value) or not FUNCTION_PATTERN.fullmatch(value):
                fail(where, f'derived value of {key!r} must be a float or the name of a function "<module>.<name>"')
            check_formula_module(value, where)
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


def parse_setup(setup: object, where: str, *, has_data: bool) -> None:
    """Check a case's ``setup``: an ``escalc`` call and at most one of ``vblocks`` and ``vconst``, each with its ``cluster``."""
    if not isinstance(setup, dict) or not setup or not set(setup) <= SETUP_KEYS:
        fail(where, f'setup must be a table whose keys are among {sorted(SETUP_KEYS)}')
    if not all(isinstance(value, dict) for key, value in setup.items() if key != 'drop'):
        fail(where, 'each setup entry except drop must be a table')
    if 'drop' in setup and not (isinstance(setup['drop'], list) and all(isinstance(i, int) and not isinstance(i, bool) for i in setup['drop'])):
        fail(where, 'setup.drop must be a list of 0-based row indices')
    if ('where' in setup or 'drop' in setup or 'rma_uni' in setup) and not has_data:
        fail(where, 'setup.where, setup.drop and setup.rma_uni need a dataset or inline data')
    if 'vblocks' in setup and 'vconst' in setup:
        fail(where, 'setup has both vblocks and vconst; give one')
    for key, allowed in (('vblocks', VBLOCKS_KEYS), ('vconst', VCONST_KEYS)):
        if key not in setup:
            continue
        if 'cluster' not in setup[key]:
            fail(where, f'setup.{key} has no cluster')
        unknown = set(setup[key]) - allowed
        missing = allowed - set(setup[key])
        if unknown or missing:
            fail(where, f'setup.{key} takes exactly {sorted(allowed)}; unknown {sorted(unknown)}, missing {sorted(missing)}')
        if not has_data:
            fail(where, f'setup.{key} needs a dataset or inline data')
    if 'subset' in setup:
        if set(setup['subset']) != SUBSET_KEYS or not all(isinstance(value, str) and value for value in setup['subset'].values()):
            fail(where, f'setup.subset takes exactly {sorted(SUBSET_KEYS)}, each a text')
        if not has_data:
            fail(where, 'setup.subset needs a dataset or inline data')
    if 'tree_r' in setup:
        spec = setup['tree_r']
        if set(spec) != TREE_R_KEYS or not all(isinstance(value, str) and value for value in spec.values()):
            fail(where, f'setup.tree_r takes exactly {sorted(TREE_R_KEYS)}, each a name')
        if not has_data:
            fail(where, 'setup.tree_r needs a dataset or inline data')
    if 'vblocks' in setup:
        columns = setup['vblocks']['columns']
        if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
            fail(where, 'setup.vblocks.columns must be a non-empty list of column names')


def parse_then(raw: object, where: str) -> Step:
    if not isinstance(raw, dict):
        fail(where, 'then must be a table')
    unknown = set(raw) - STEP_KEYS
    if unknown:
        fail(where, f'then: unknown key(s) {sorted(unknown)}; only function and args')
    function = raw.get('function')
    if not isinstance(function, str) or not FUNCTION_PATTERN.fullmatch(function) or is_formula(function):
        fail(where, f'then: function {function!r} must be a metafopy function name')
    args = raw.get('args', {})
    if not isinstance(args, dict):
        fail(where, 'then: args must be a table')
    return Step(function=function, args=dict(args))


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
        fail(name, f'case {ident}: function {function!r} must be a metafopy function name or "<module>.<name>"')
    if is_formula(function):
        check_formula_module(function, f'{name}: case {ident}')
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
    data = raw.get('data')
    if data is not None and not isinstance(data, dict):
        fail(name, f'case {ident}: data must be a table of columns')
    if data is not None and 'dataset' in raw:
        fail(name, f'case {ident}: give a dataset or inline data, not both')
    if setup is not None:
        parse_setup(setup, where, has_data=data is not None or 'dataset' in raw)
    then = parse_then(raw['then'], where) if 'then' in raw else None
    slow = raw.get('slow', False)
    if not isinstance(slow, bool):
        fail(name, f'case {ident}: slow must be true or false')
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
        then=then,
        slow=slow,
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


def load_derived_required(directory: pathlib.Path | None = None) -> list[str]:
    """The ``derived_required`` prefixes of every case file: an item starting with one needs a ``derived`` case, a ``published`` one alone not enough."""
    out: list[str] = []
    for path in case_files(directory):
        raw = parse_document(path.read_text(encoding='utf-8'), path.name).get('derived_required', [])
        if not isinstance(raw, list) or not all(isinstance(v, str) and v.strip() for v in raw):
            fail(path.name, f'derived_required must be a list of non-empty strings, got {raw!r}')
        out += raw
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
    """The case's data: the bundled dataset or the inline columns, the rows ``where`` keeps and ``drop`` leaves, the ``escalc`` of ``setup``, then the
    rows ``subset`` keeps. ``where`` or ``drop`` that does not fit the data raises ``ValueError`` naming file and case."""
    if case.dataset is not None:
        frame = metafopy.datasets.load(case.dataset)
    elif case.data is not None:
        frame = pandas.DataFrame(case.data)
    else:
        return None
    where = f'{case.file}: case {case.id}'
    if case.setup is not None and 'where' in case.setup:
        for column, value in case.setup['where'].items():
            if column not in frame.columns:
                fail(where, f'setup.where column {column!r} is not in the data')
            frame = frame[frame[column] == value]
        frame = frame.reset_index(drop=True)
    if case.setup is not None and 'drop' in case.setup:
        bad = [i for i in case.setup['drop'] if not 0 <= i < len(frame)]
        if bad:
            fail(where, f'setup.drop index {bad} is outside the {len(frame)} rows')
        frame = frame.drop(index=case.setup['drop']).reset_index(drop=True)
    if case.setup is not None and 'escalc' in case.setup:
        frame = metafopy.escalc(data=frame, **case.setup['escalc'])
    if case.setup is not None and 'subset' in case.setup:
        frame = frame[frame[case.setup['subset']['column']] == case.setup['subset']['equals']].reset_index(drop=True)
    return frame


def build_v(frame: pandas.DataFrame, setup: dict[str, typing.Any], where: str) -> FloatArray:
    """The ``V`` matrix a ``vblocks`` or ``vconst`` setup describes, rows in data order, computed with numpy and pandas alone.

    ``vblocks`` is block-diagonal: the block of a cluster has one row per row of the cluster and one column per name in ``columns``, so that entry
    ``[i, l]`` is ``frame[columns[l]]`` at the cluster's ``i``-th row. ``vconst`` puts ``vi`` on the diagonal and ``rho * sqrt(vi[i] * vi[l])`` between
    two rows of a cluster.
    """
    spec = setup['vblocks'] if 'vblocks' in setup else setup['vconst']
    labels = frame[spec['cluster']].to_numpy()
    k = len(frame)
    matrix = numpy.zeros((k, k), dtype=numpy.float64)
    for label in pandas.unique(labels).tolist():
        rows = numpy.flatnonzero(labels == label)
        if 'vblocks' in setup:
            columns = spec['columns']
            if len(rows) != len(columns):
                raise ValueError(f'{where}: cluster {label!r} has {len(rows)} rows but vblocks names {len(columns)} columns')
            block = frame[columns].to_numpy(dtype=numpy.float64)[rows]
            if not numpy.allclose(block, block.T, rtol=SYMMETRY_TOLERANCE, atol=SYMMETRY_TOLERANCE):
                raise ValueError(f'{where}: the block of cluster {label!r} is not symmetric')
        else:
            vi = frame[spec['vi']].to_numpy(dtype=numpy.float64)[rows]
            block = spec['rho'] * numpy.sqrt(numpy.outer(vi, vi))
            numpy.fill_diagonal(block, vi)
        matrix[numpy.ix_(rows, rows)] = block
    return matrix


def build_tree_r(tree: str) -> pandas.DataFrame:
    """The covariance matrix of a Brownian-motion trait along a bundled tree: entry ``(i, j)`` is the length of the edges that the root-to-tip paths
    of tips ``i`` and ``j`` share, computed with numpy from the edge table alone (Hadfield and Nakagawa 2010), rows and columns labelled by the tips."""
    edges = metafopy.datasets.load_tree(tree).edges
    parent = dict(zip(edges['child'], edges['parent'], strict=True))
    length = dict(zip(edges['child'], edges['length'], strict=True))
    column = {child: position for position, child in enumerate(edges['child'])}
    tips = sorted(set(edges['child']) - set(edges['parent']))
    paths = numpy.zeros((len(tips), len(column)))
    for row, tip in enumerate(tips):
        node = tip
        while node in parent:
            paths[row, column[node]] = length[node]
            node = parent[node]
    shared = (paths > 0).astype(numpy.float64)
    labels = edges.set_index('child').loc[tips, 'child_label'].astype(str).tolist()
    return pandas.DataFrame((shared * numpy.array([length[child] for child in edges['child']])) @ shared.T, index=labels, columns=labels)


def is_formula(name: str) -> bool:
    """A dotted function name is a formula in a module of ``FORMULA_MODULES``; a metafopy function name has no dot."""
    return '.' in name


def check_formula_module(name: str, where: str) -> None:
    """Refuse a formula whose module is not in ``FORMULA_MODULES``."""
    module_name = name.partition('.')[0]
    if module_name not in FORMULA_MODULES:
        fail(where, f'function {name!r} must be a metafopy function name or "<module>.<name>" with a module in {list(FORMULA_MODULES)}')


def resolve_formula(name: str, where: str) -> collections.abc.Callable[..., typing.Any]:
    module_name, _, function = name.partition('.')
    check_formula_module(name, where)
    func = getattr(FORMULA_MODULE_OBJECTS[module_name], function, None)
    if not callable(func):
        raise LookupError(f'{where}: {module_name}.py has no function {function!r}')
    return typing.cast(collections.abc.Callable[..., typing.Any], func)


def formula_value(case: Case, name: str, frame: pandas.DataFrame | None) -> typing.Any:
    """Call a formula with the ``args`` of the case that it declares, a column name replaced by the column."""
    func = resolve_formula(name, case.id)
    kwargs: dict[str, typing.Any] = {}
    for param, signature in inspect.signature(func).parameters.items():
        if param == 'data' and frame is not None:
            kwargs[param] = frame
            continue
        if param not in case.args:
            continue
        value = case.args[param]
        # a parameter annotated ``str`` takes the text as it is, so ``predictor = "vi"`` is not read as the column ``vi``
        if isinstance(value, str) and signature.annotation is not str and frame is not None and value in frame.columns:
            value = frame[value].to_numpy(dtype=numpy.float64)
        kwargs[param] = value
    return func(**kwargs)


def run_case(case: Case) -> typing.Any:
    """Load the dataset, apply ``setup``, call ``metafopy.<function>(**args)`` (``<module>.<name>`` for a formula), then apply ``then`` to the fit.

    ``setup`` is an ``escalc`` call, ``where`` and ``drop`` row filters, for ``rma_uni`` the model fitted on ``yi`` and ``vi`` and passed as the first
    argument in place of ``data``, for ``vblocks`` or ``vconst`` the ``V`` passed to the call, and for ``tree_r`` the ``R`` passed to it (the
    unscaled tree covariance matrix of the named factor). The result of ``then`` is returned.
    """
    frame = case_frame(case)
    if is_formula(case.function):
        return formula_value(case, case.function, frame)
    func = getattr(metafopy, case.function, None)
    if not callable(func):
        raise LookupError(f'{case.id}: metafopy has no function {case.function!r}')
    kwargs = dict(case.args)
    if frame is not None and case.setup is not None and 'rma_uni' in case.setup:
        fit = metafopy.rma_uni(yi='yi', vi='vi', data=frame, **case.setup['rma_uni'])
        return apply_then(case, func(fit, **kwargs))
    if frame is not None:
        kwargs['data'] = frame
        if case.setup is not None and ('vblocks' in case.setup or 'vconst' in case.setup):
            kwargs['V'] = build_v(frame, case.setup, f'{case.file}: case {case.id}')
        if case.setup is not None and 'tree_r' in case.setup:
            kwargs['R'] = {case.setup['tree_r']['factor']: build_tree_r(case.setup['tree_r']['tree'])}
    return apply_then(case, func(**kwargs))


def apply_then(case: Case, result: typing.Any) -> typing.Any:
    """The result of the case's ``then`` step on ``result``, or ``result`` when the case has none."""
    if case.then is None:
        return result
    follow = getattr(metafopy, case.then.function, None)
    if not callable(follow):
        raise LookupError(f'{case.file}: case {case.id}: metafopy has no follow-on function {case.then.function!r}')
    return follow(result, **case.then.args)


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
    frame = case_frame(case) if any(isinstance(v, str) and is_formula(v) for v in case.expect.values()) else None
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
    if case.slow:
        marks.append(pytest.mark.slow)
    return pytest.param(case, id=case.id, marks=marks)


def case_params(kind: str) -> list[typing.Any]:
    """Cases of one kind as pytest params, id = case id; a ``mismatch`` case is marked ``xfail(strict=True, reason=...)``."""
    return [case_param(c) for c in load_cases(kind)]
