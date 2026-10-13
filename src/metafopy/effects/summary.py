"""Standard errors, test statistics and confidence intervals for escalc output (metafor's ``summary.escalc``).

A port of ``summary.escalc()`` from metafor 5.2-1 (``R/summary.escalc.r``);
see https://wviechtb.github.io/metafor/reference/print.escalc.html. It works on
the :class:`pandas.DataFrame` returned by :func:`metafopy.escalc`, whose
metafor attributes live in ``DataFrame.attrs`` (``yi_names``, ``vi_names``,
``measure``, ``ni``, ...).

Differences from R that follow Python conventions:

- ``out_names`` defaults to ``('sei', 'zi', 'pval', 'ci_lb', 'ci_ub')`` and
  any strings are accepted (R makes them syntactically valid names);
- like R's ``mapply()``, ``transf`` is called once per value; keyword
  arguments in ``**kwargs`` that have one entry per row are passed
  elementwise, everything else (scalars, mappings such as ``targs``) as is;
- the result keeps the input's row index.
"""

import collections.abc
import typing

import numpy
import numpy.typing
import pandas
import scipy.special
import scipy.stats

import metafopy.errors

__all__ = ['summary_escalc']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
ATTR_KEYS = ('sei_names', 'zi_names', 'pval_names', 'ci_lb_names', 'ci_ub_names')


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def level_to_alpha(level: float) -> float:
    """metafor's ``.level``: a confidence level in percent (or as a proportion) to alpha.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if level > 100 or level < 0:
        raise metafopy.errors.InputError("Argument 'level' must be between 0 and 100.")
    if level == 0:
        return 1.0
    if level >= 1:
        return (100 - level) / 100
    return 1 - level if level > 0.5 else level


def names(value: typing.Any) -> list[str]:
    """An attribute holding one name or several as a list (``None`` for none)."""
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def union(first: str, rest: typing.Any) -> list[str]:
    """R's ``union(first, rest)`` for character vectors."""
    out = [first]
    for name in names(rest):
        if name not in out:
            out.append(name)
    return out


def mapply(transf: collections.abc.Callable[..., typing.Any], values: FloatArray, kwargs: collections.abc.Mapping[str, typing.Any]) -> FloatArray:
    """``mapply(transf, values, ...)``: one call per value, with per-row keyword arguments taken elementwise.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    n = values.size
    per_row: dict[str, typing.Any] = {}
    shared: dict[str, typing.Any] = {}
    for key, val in kwargs.items():
        if val is None or isinstance(val, (str, collections.abc.Mapping)) or numpy.ndim(val) == 0:
            shared[key] = val
        else:
            arr = numpy.asarray(val)
            if arr.size == n:
                per_row[key] = arr.reshape(-1)
            elif arr.size == 1:
                shared[key] = arr.reshape(-1)[0]
            else:
                raise metafopy.errors.InputError(f"Argument '{key}' must have length 1 or one value per row.")
    out = numpy.empty(n, dtype=numpy.float64)
    for i in range(n):
        res = numpy.asarray(transf(values[i], **shared, **{k: v[i] for k, v in per_row.items()}), dtype=numpy.float64).reshape(-1)
        if res.size != 1:
            raise metafopy.errors.InputError("The 'transf' function must return a single value for each value it is given.")
        out[i] = res[0]
    return out


def summary_escalc(
    object: pandas.DataFrame,
    out_names: collections.abc.Sequence[str] = ('sei', 'zi', 'pval', 'ci_lb', 'ci_ub'),
    var_names: collections.abc.Sequence[str] | None = None,
    H0: numpy.typing.ArrayLike = 0,
    append: bool = True,
    replace: bool = True,
    level: float = 95,
    olim: collections.abc.Sequence[float] | None = None,
    digits: int | collections.abc.Sequence[int] | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    **kwargs: typing.Any,
) -> pandas.DataFrame:
    """Add standard errors, z-values, p-values and confidence intervals to escalc output.

    ``sei = sqrt(vi)``, ``zi = (yi - H0)/sei``, two-sided ``pval`` and the
    Wald-type interval ``yi -/+ z_crit * sei`` at the given ``level``. With
    ``transf``, ``yi`` and the interval bounds are transformed (the interval
    is re-ordered when ``transf`` is decreasing) and ``vi``, ``sei``, ``zi``
    and ``pval`` are dropped; ``kwargs`` are passed on to ``transf``.
    ``olim`` clips ``yi`` and the bounds to an interval (``zi`` and ``pval``
    are based on the unclipped values).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    alpha = level_to_alpha(level)
    crit = float(scipy.stats.norm.isf(alpha / 2))
    if len(out_names) != 5:
        raise metafopy.errors.InputError("Argument 'out_names' must be of length 5.")
    out_names = list(out_names)

    attrs = object.attrs
    if var_names is None:
        if attrs.get('yi_names'):
            yi_name = names(attrs['yi_names'])[0]
        elif 'yi' in object.columns:
            yi_name = 'yi'
        else:
            raise metafopy.errors.InputError("Cannot determine the name of the 'yi' variable.")
        if attrs.get('vi_names'):
            vi_name = names(attrs['vi_names'])[0]
        elif 'vi' in object.columns:
            vi_name = 'vi'
        else:
            raise metafopy.errors.InputError("Cannot determine the name of the 'vi' variable.")
    else:
        if len(var_names) != 2:
            raise metafopy.errors.InputError("Argument 'var_names' must be of length 2.")
        yi_name, vi_name = var_names[0], var_names[1]
    for name in (yi_name, vi_name):
        if name not in object.columns:
            raise metafopy.errors.InputError(f"Cannot find variable '{name}' in the data frame.")

    yi = object[yi_name].to_numpy(dtype=numpy.float64, copy=True)
    vi_in = object[vi_name].to_numpy(dtype=numpy.float64, copy=True)
    vi: FloatArray | None = vi_in
    k = yi.size
    h0 = numpy.asarray(H0, dtype=numpy.float64).reshape(-1)
    if h0.size == 1:
        h0 = numpy.full(k, h0[0])

    with numpy.errstate(invalid='ignore', divide='ignore'):
        se = numpy.sqrt(vi_in)
        z = (yi - h0) / se
    sei: FloatArray | None = se
    zi: FloatArray | None = z
    pval: FloatArray | None = 2 * scipy.special.ndtr(-numpy.abs(z))
    if transf is not None:
        ci_lb = mapply(transf, yi - crit * se, kwargs)
        ci_ub = mapply(transf, yi + crit * se, kwargs)
        yi = mapply(transf, yi, kwargs)
        vi = sei = zi = pval = None
    else:
        ci_lb = yi - crit * se
        ci_ub = yi + crit * se

    # make sure the bounds are in increasing order (pairs with a NaN are left alone)
    flip = ci_lb > ci_ub
    ci_lb, ci_ub = numpy.where(flip, ci_ub, ci_lb), numpy.where(flip, ci_lb, ci_ub)

    if olim is not None:
        if len(olim) != 2:
            raise metafopy.errors.InputError("Argument 'olim' must be of length 2.")
        lo, hi = sorted(float(v) for v in olim)
        yi, ci_lb, ci_ub = (numpy.where(v < lo, lo, numpy.where(v > hi, hi, v)) for v in (yi, ci_lb, ci_ub))

    x = object.copy()
    x[yi_name] = yi
    if vi is None:
        x = x.drop(columns=vi_name)

    values = (sei, zi, pval, ci_lb, ci_ub)
    if append:
        dat = x
        for name, val in zip(out_names, values, strict=True):
            if val is None:
                # R assigns NULL: the column is removed (replace=TRUE) or left as it is
                if replace and name in dat.columns:
                    dat = dat.drop(columns=name)
            elif replace or name not in dat.columns:
                dat[name] = val
            else:
                col = dat[name].to_numpy(dtype=numpy.float64, copy=True)
                miss = numpy.isnan(col)
                col[miss] = val[miss]
                dat[name] = col
    elif transf is not None:
        dat = pandas.DataFrame({yi_name: yi, out_names[3]: ci_lb, out_names[4]: ci_ub}, index=object.index)
    else:
        dat = pandas.DataFrame(dict(zip([yi_name, vi_name, *out_names], [yi, vi, *values], strict=True)), index=object.index)

    # the escalc attributes stay with an untransformed yi (in R they are attributes of the yi vector)
    new_attrs: dict[str, typing.Any] = {}
    if transf is None:
        new_attrs.update({key: attrs[key] for key in ('measure', 'slab', 'ni') if key in attrs})
    new_attrs['digits'] = digits if digits is not None else attrs.get('digits')
    new_attrs['yi_names'] = union(yi_name, attrs.get('yi_names'))
    new_attrs['vi_names'] = union(vi_name, attrs.get('vi_names'))
    for key, name in zip(ATTR_KEYS, out_names, strict=True):
        new_attrs[key] = union(name, attrs.get(key))
    dat.attrs = {}
    dat.attrs.update(new_attrs)
    return dat
