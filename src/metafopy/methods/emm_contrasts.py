"""Contrast coefficients for ``contrast()`` (emmeans' ``*.emmc`` functions).

R source: https://github.com/rvlenth/emmeans/blob/master/R/contrast.R and ``R/emmc-fcns.R``;
reference: https://rvlenth.github.io/emmeans/reference/contrast.html.
Indices (``ref``, ``exclude``, ``include``) are 0-based; level labels are accepted as well.
"""

import collections.abc
import dataclasses

import numpy
import numpy.typing

import metafopy.errors

__all__ = ['METHODS', 'ContrastSet', 'build_contrasts']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Key = int | str | collections.abc.Sequence[int | str]

METHODS = (
    'consec',
    'del_eff',
    'dunnett',
    'eff',
    'identity',
    'mean_chg',
    'pairwise',
    'poly',
    'revpairwise',
    'trt_vs_ctrl',
    'trt_vs_ctrl1',
    'trt_vs_ctrlk',
    'tukey',
)


@dataclasses.dataclass(frozen=True, slots=True)
class ContrastSet:
    """Coefficients (one column per contrast, one row per level) with the family information emmeans keeps as attributes."""

    names: tuple[str, ...]
    coef: FloatArray
    desc: str
    adjust: str
    est_type: str | None = None
    fam_size: int | None = None


def indices(levs: collections.abc.Sequence[str], key: Key) -> list[int]:
    """0-based positions of ``key`` (an index, a level label or a sequence of them) among the levels.

    Raises:
        metafopy.errors.InputError: a label is not one of the levels.
    """
    keys = [key] if isinstance(key, int | str) else list(key)
    out: list[int] = []
    for k in keys:
        if isinstance(k, str):
            if k not in levs:
                raise metafopy.errors.InputError(f'{k!r} is not one of the levels: {", ".join(levs)}.')
            out.append(list(levs).index(k))
        else:
            out.append(k % len(levs) if k < 0 else k)
    return out


def excluded(levs: collections.abc.Sequence[str], exclude: Key | None, include: Key | None) -> list[int]:
    if include is not None:
        keep = set(indices(levs, include))
        return [i for i in range(len(levs)) if i not in keep]
    return [] if exclude is None else indices(levs, exclude)


def frame(columns: list[tuple[str, FloatArray]], desc: str, adjust: str, **extra: object) -> ContrastSet:
    names = tuple(name for name, _ in columns)
    coef = numpy.column_stack([col for _, col in columns]) if columns else numpy.zeros((0, 0))
    est_type = extra.get('est_type')
    fam_size = extra.get('fam_size')
    return ContrastSet(names, coef, desc, adjust, est_type if isinstance(est_type, str) else None, fam_size if isinstance(fam_size, int) else None)


def pairwise(levs: collections.abc.Sequence[str], exclude: list[int], reverse: bool) -> ContrastSet:
    k = len(levs)
    cols: list[tuple[str, FloatArray]] = []
    pairs = (
        [(i, j) for i in range(1, k) if i not in exclude for j in range(i) if j not in exclude]
        if reverse
        else [(i, j) for i in range(k - 1) if i not in exclude for j in range(i + 1, k) if j not in exclude]
    )
    for i, j in pairs:
        con = numpy.zeros(k)
        con[i] = 1
        con[j] = -1
        cols.append((f'{levs[i]} - {levs[j]}', con))
    return frame(cols, 'pairwise differences', 'tukey', est_type='pairs', fam_size=k - len(exclude))


def trt_vs_ctrl(levs: collections.abc.Sequence[str], ref: list[int], reverse: bool, exclude: list[int]) -> ContrastSet:
    k = len(levs)
    if not ref or min(ref) < 0 or max(ref) >= k:
        raise metafopy.errors.InputError("In trt_vs_ctrl, 'ref' levels are out of range.")
    if set(exclude) & set(ref):
        raise metafopy.errors.InputError("'exclude' set cannot overlap with 'ref'.")
    cnm = levs[ref[0]] if len(ref) == 1 else f'avg({",".join(levs[r] for r in ref)})'
    templ = numpy.zeros(k)
    templ[ref] = -1 / len(ref)
    cols: list[tuple[str, FloatArray]] = []
    for i in range(k):
        if i in ref or i in exclude:
            continue
        con = templ.copy()
        con[i] = 1
        cols.append((f'{cnm} - {levs[i]}' if reverse else f'{levs[i]} - {cnm}', -con if reverse else con))
    return frame(cols, 'differences from control', 'dunnettx', fam_size=k - len(exclude) if exclude else None)


def eff(levs: collections.abc.Sequence[str], exclude: list[int], wts: collections.abc.Sequence[float] | None, del_eff: bool) -> ContrastSet:
    k = len(levs)
    w = numpy.ones(k) if wts is None else numpy.asarray(wts, dtype=numpy.float64).copy()
    if exclude and w.size == k - len(exclude):
        tmp = numpy.zeros(k)
        tmp[[i for i in range(k) if i not in exclude]] = w
        w = tmp
    if w.size != k:
        raise metafopy.errors.InputError("length of 'wts' must equal the number of levels or the number of included levels.")
    w[exclude] = 0
    w = w / w.sum()
    cols: list[tuple[str, FloatArray]] = []
    for i in (i for i in range(k) if i not in exclude):
        con = -w.copy()
        con[i] += 1
        cols.append((f'{levs[i]} effect', con / con[i] if del_eff else con))
    return frame(cols, 'differences from mean of others' if del_eff else 'differences from grand mean', 'fdr', fam_size=k - len(exclude) if exclude else None)


def consec(levs: collections.abc.Sequence[str], exclude: list[int], reverse: bool) -> ContrastSet:
    sgn = -1.0 if reverse else 1.0
    active = [i for i in range(len(levs)) if i not in exclude]
    nms = [levs[i] for i in active]
    cols: list[tuple[str, FloatArray]] = []
    for i in range(len(active) - 1):
        tmp = numpy.zeros(len(levs))
        tmp[active[i]] = -sgn
        tmp[active[i + 1]] = sgn
        cols.append((f'{nms[i]} - {nms[i + 1]}' if reverse else f'{nms[i + 1]} - {nms[i]}', tmp))
    return frame(cols, 'changes between consecutive levels', 'mvt', fam_size=len(levs) - len(exclude) if exclude else None)


def mean_chg(levs: collections.abc.Sequence[str], exclude: list[int], reverse: bool) -> ContrastSet:
    sgn = -1.0 if reverse else 1.0
    active = [i for i in range(len(levs)) if i not in exclude]
    k = len(active)
    nms = [levs[i] for i in active]
    cols: list[tuple[str, FloatArray]] = []
    for i in range(1, k):
        tmp = numpy.zeros(len(levs))
        tmp[active] = numpy.concatenate([numpy.full(i, -sgn / i), numpy.full(k - i, sgn / (k - i))])
        cols.append((f'{nms[i - 1]}|{nms[i]}', tmp))
    return frame(cols, 'mean after minus mean before', 'mvt', fam_size=len(levs) - len(exclude) if exclude else None)


def poly(levs: collections.abc.Sequence[str], max_degree: int | None) -> ContrastSet:
    names = ['linear', 'quadratic', 'cubic', 'quartic', *[f'degree {d}' for d in range(5, 21)]]
    k = len(levs)
    degree = min(20, min(6, k - 1) if max_degree is None else max_degree)
    x = numpy.arange(1, k + 1, dtype=numpy.float64)
    vander = numpy.vander(x - x.mean(), degree + 1, increasing=True)
    q, r = numpy.linalg.qr(vander)
    cols: list[tuple[str, FloatArray]] = []
    for j in range(1, degree + 1):
        con = q[:, j] * r[j, j]
        con = con / numpy.linalg.norm(con)
        pos = con > 0.01
        con = con / con[pos].min()
        z = numpy.abs(con - numpy.round(con)).max()
        while z > 0.05:
            con = con / z
            z = numpy.abs(con - numpy.round(con)).max()
        cols.append((names[j - 1], numpy.round(con)))
    return frame(cols, 'polynomial contrasts', 'none')


def identity(levs: collections.abc.Sequence[str], exclude: list[int]) -> ContrastSet:
    keep = [i for i in range(len(levs)) if i not in exclude]
    cols = [(levs[i], numpy.eye(len(levs))[:, i]) for i in keep]
    return frame(cols, 'Identity', 'none', fam_size=len(keep))


def custom(levs: collections.abc.Sequence[str], spec: collections.abc.Mapping[str, collections.abc.Sequence[float]]) -> ContrastSet:
    cols: list[tuple[str, FloatArray]] = []
    for name, coefs in spec.items():
        con = numpy.asarray(coefs, dtype=numpy.float64)
        if con.shape != (len(levs),):
            raise metafopy.errors.InputError(f'Nonconforming number of contrast coefficients for {name!r}: need {len(levs)}.')
        cols.append((str(name), con))
    return frame(cols, 'custom', 'none')


def build_contrasts(
    method: str | collections.abc.Mapping[str, collections.abc.Sequence[float]],
    levs: collections.abc.Sequence[str],
    *,
    ref: Key | None = None,
    exclude: Key | None = None,
    include: Key | None = None,
    reverse: bool = False,
    wts: collections.abc.Sequence[float] | None = None,
    max_degree: int | None = None,
) -> ContrastSet:
    """The contrast coefficients for the levels ``levs``, for one of :data:`METHODS` or a mapping of names to coefficient vectors.

    Raises:
        metafopy.errors.InputError: the method is unknown or an argument does not fit the levels.
    """
    if isinstance(method, collections.abc.Mapping):
        return custom(levs, method)
    method = method.replace('.', '_')  # emmeans spells them trt.vs.ctrl and del.eff
    excl = excluded(levs, exclude, include)
    k = len(levs)
    match method:
        case 'pairwise' | 'tukey':
            return pairwise(levs, excl, reverse)
        case 'revpairwise':
            return pairwise(levs, excl, True)
        case 'trt_vs_ctrl' | 'trt_vs_ctrl1' | 'dunnett':
            return trt_vs_ctrl(levs, indices(levs, 0 if ref is None else ref), reverse, excl)
        case 'trt_vs_ctrlk':
            return trt_vs_ctrl(levs, indices(levs, k - 1 if ref is None else ref), reverse, excl)
        case 'eff':
            return eff(levs, excl, wts, False)
        case 'del_eff':
            return eff(levs, excl, wts, True)
        case 'consec':
            return consec(levs, excl, reverse)
        case 'mean_chg':
            return mean_chg(levs, excl, reverse)
        case 'poly':
            return poly(levs, max_degree)
        case 'identity':
            return identity(levs, excl)
    raise metafopy.errors.InputError(f'Contrast function {method!r} not found; use one of {", ".join(METHODS)}.')
