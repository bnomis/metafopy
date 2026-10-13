"""Fail-safe N (metafor's ``fsn``).

Rosenthal (1979), doi:10.1037/0033-2909.86.3.638; Orwin (1983), doi:10.3102/10769986008002157;
Rosenberg (2005), doi:10.1111/j.0014-3820.2005.tb01004.x; Viechtbauer's general approach based on
a random-effects model. See https://wviechtb.github.io/metafor/reference/fsn.html.
"""

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.bias._inputs import data_of, fp, fx, vector_inputs
from metafopy.core.rng import RRandom, get_rng, set_seed
from metafopy.core.zeroin import uniroot
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult, rma_uni

__all__ = ['FsnResult', 'fsn']

type FloatArray = NDArray[np.float64]
type FsnType = Literal['Rosenthal', 'Orwin', 'Rosenberg', 'General', 'Binomial']

_EPS = sys.float_info.epsilon


@dataclass(frozen=True, slots=True)
class FsnResult:
    type: str
    fsnum: float
    est: float
    tau2: float
    pval: float
    alpha: float
    target: float
    method: str | None
    est_fsn: float
    tau2_fsn: float
    pval_fsn: float
    ub_sign: str
    digits: int = 4

    @property
    def meanes(self) -> float:
        return self.est

    def summary(self, digits: int | None = None) -> str:
        d = self.digits if digits is None else digits
        lines = ['', f'Fail-safe N Calculation Using the {self.type} Approach', '']
        alpha = f'{round(self.alpha, d):.7g}'
        if self.type in ('Rosenthal', 'Binomial'):
            lines += [f'Observed Significance Level: {fp(self.pval, d)}', f'Target Significance Level:   {alpha}']
        elif self.type == 'Orwin':
            lines += [f'Average Effect Size: {fx(self.est, d)}', f'Target Effect Size:  {fx(self.target, d)}']
        elif self.type == 'Rosenberg':
            left = ' ' if self.est < 0 else ''
            lines += [
                f'Average Effect Size:         {fx(self.est, d, left)}',
                f'Observed Significance Level: {left}{fp(self.pval, d)}',
                f'Target Significance Level:   {left}{alpha}',
            ]
        else:
            left = ' ' if self.est < 0 else ''
            right = ' ' if self.est_fsn < 0 else ''
            file_drawer = self.fsnum > 0
            line = f'Average Effect Size:         {fx(self.est, d, left)}'
            if file_drawer:
                line += f' (with file drawer: {fx(self.est_fsn, d, right)})'
            lines.append(line)
            if self.method not in FIXED_EFFECTS:
                line = f'Amount of Heterogeneity:     {fx(self.tau2, d, left)}'
                if file_drawer:
                    line += f' (with file drawer: {fx(self.tau2_fsn, d, right)})'
                lines.append(line)
            line = f'Observed Significance Level: {left}{fp(self.pval, d)}'
            if file_drawer:
                line += f' (with file drawer: {right}{fp(self.pval_fsn, d)})'
            lines.append(line)
            if math.isnan(self.target):
                lines.append(f'Target Significance Level:   {left}{alpha}')
            else:
                lines.append(f'Target Effect Size:          {fx(self.target, d, left)}')
        num = 'Inf' if math.isinf(self.fsnum) else f'{int(self.fsnum)}'
        return '\n'.join([*lines, '', f'Fail-safe N: {self.ub_sign}{num}', ''])

    def __str__(self) -> str:
        return self.summary()


def rnd_fsn(x: float) -> float:
    if math.isfinite(x) and abs(x - round(x)) >= _EPS**0.5:
        return float(math.ceil(x))
    return float(round(x)) if math.isfinite(x) else x


def _scale(x: FloatArray, k: int) -> FloatArray:
    if k == 0:
        return x
    if k == 1:
        return np.zeros(1)
    return (x - x.mean()) / x.std(ddof=1)


def _fsn_fisher(fsnum: float, pi: FloatArray, alpha: float) -> float:
    n = int(fsnum)
    x2 = -2 * float(np.sum(np.log(np.concatenate([pi, np.full(n, 0.5)]))))
    return float(stats.chi2.sf(x2, 2 * (pi.shape[0] + fsnum))) - alpha


def _wmean(x: FloatArray, w: FloatArray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def fsn(
    x: RMAResult | ArrayLike | str,
    vi: ArrayLike | str | None = None,
    *,
    sei: ArrayLike | str | None = None,
    subset: ArrayLike | None = None,
    data: pd.DataFrame | None = None,
    type: FsnType | None = None,
    alpha: float = 0.05,
    target: float | None = None,
    method: str | None = None,
    exact: bool = False,
    digits: int | None = None,
    pool: Literal['stouffer', 'fisher'] = 'stouffer',
    mumiss: float = 0,
    interval: tuple[float, float] | None = None,
    maxint: float = 1e7,
    tol: float = _EPS**0.25,
    maxiter: int = 1000,
    tau2: float | None = None,
    test: str | None = None,
    weighted: bool | None = None,
    seed: int | None = None,
    rng: RRandom | None = None,
) -> FsnResult:
    """Fail-safe N calculation.

    ``type='General'`` (the only choice for a fitted model) adds studies with an average effect of
    ``mumiss`` until the model's p-value exceeds ``alpha`` (or the estimate reaches ``target``). With
    ``exact=True`` a model is refitted to data in which the file-drawer studies are random normal
    deviates, standardised to mean ``mumiss`` and the expected variance; those come from the R-compatible
    generator (:func:`metafopy.set_seed`, or ``seed`` / ``rng`` here).
    """
    gen = rng if rng is not None else (set_seed(seed) if seed is not None else get_rng())
    typ: str
    method_given = method
    is_model = isinstance(x, RMAResult)
    if isinstance(x, RMAResult):
        if not x.int_only:
            raise ValueError('Method only applicable to models without moderators.')
        if type is not None and type != 'General':
            warnings.warn("Setting type='General' when using fsn() on a model object.", stacklevel=2)
        typ = 'General'
        if x.weights is not None:
            raise ValueError('Cannot use function on models with custom weights.')
        yi, v, _ = data_of(x)
    else:
        typ_in = 'Rosenthal' if type is None else type
        options = ['rosenthal', 'binomial', 'orwin', 'rosenberg', 'general']
        match = [o for o in options if o.startswith(typ_in.lower())]
        if not match:
            raise ValueError("Unknown 'type' specified.")
        typ = match[0].capitalize()
        v_in: Any = vi
        if typ == 'Orwin' and weighted is False:
            v_in, sei = 1.0, None
        elif typ == 'Orwin' and weighted is None and vi is None and sei is None:
            v_in = 1.0
        if isinstance(v_in, float) and v_in == 1.0:
            ones = np.ones(np.atleast_1d(np.asarray(x, dtype=np.float64)).shape[0])
            yi, v_opt, _ = vector_inputs(x, ones, None, subset=subset, data=data, what='the calculation')
        else:
            yi, v_opt, _ = vector_inputs(
                x, v_in, sei, subset=subset, data=data, what='the calculation', vi_required=typ in ('Rosenthal', 'Rosenberg', 'General')
            )
        if v_opt is None:
            if typ != 'Binomial':
                raise ValueError("Must specify the 'vi' or 'sei' argument.")
            v_opt = np.ones_like(yi)
        v = v_opt
    if np.any(v <= 0):
        raise ValueError('Cannot use function when there are non-positive sampling variances in the data.')
    k = yi.shape[0]
    if k == 1:
        raise ValueError('Stopped because k = 1.')
    d = 4 if digits is None else digits
    ival = (0.0, k * 50.0) if interval is None else interval

    est = tau2_out = pval = est_fsn = tau2_fsn = pval_fsn = math.nan
    ub_sign = ''
    meth_out: str | None = None
    tgt = math.nan if target is None else target
    fsnum: float

    if typ == 'Rosenthal' and pool == 'stouffer':
        zi = yi / np.sqrt(v)
        z_avg = abs(float(np.sum(zi)) / math.sqrt(k))
        pval = float(stats.norm.sf(z_avg))
        fsnum = rnd_fsn(max(0.0, k * (z_avg / float(stats.norm.isf(alpha))) ** 2 - k))
        tgt = math.nan
    elif typ == 'Rosenthal':
        zi = yi / np.sqrt(v)
        pi = stats.norm.sf(np.abs(zi))
        pval = _fsn_fisher(0, pi, 0.0)
        if pval >= alpha:
            fsnum = 0.0
        else:
            fsnum = uniroot(lambda n: _fsn_fisher(n, pi, alpha), ival[0], ival[1], tol=tol, maxiter=maxiter, extend_int='upX').root
        fsnum = rnd_fsn(fsnum)
        tgt = math.nan
    elif typ == 'Binomial':
        kpos = int(np.sum(yi > 0))
        pval = float(stats.binomtest(kpos, k).pvalue)
        fsnum = 0.0
        if pval < alpha:
            pnew = pval
            while pnew < alpha:
                fsnum += 2
                pnew = float(stats.binomtest(kpos + int(fsnum / 2), k + int(fsnum)).pvalue)
        fsnum = rnd_fsn(fsnum)
        tgt = math.nan
    elif typ == 'Orwin':
        est = _wmean(yi, 1 / v)
        if target is None:
            tgt = est / 2
        if tgt == 0:
            fsnum = math.inf
        else:
            if np.sign(tgt) != np.sign(est):
                tgt = -1 * tgt
            fsnum = max(0.0, k * (est - tgt) / tgt)
        fsnum = rnd_fsn(fsnum)
    elif typ == 'Rosenberg':
        wi = 1 / v
        est = _wmean(yi, wi)
        zval = est / math.sqrt(1 / float(np.sum(wi)))
        pval = float(2 * stats.norm.sf(abs(zval)))
        vt = 1 / float(np.mean(1 / v))
        fsnum = rnd_fsn(max(0.0, ((float(np.sum(wi * yi)) / float(stats.norm.isf(alpha / 2))) ** 2 - float(np.sum(wi))) * vt))
        tgt = math.nan
    else:
        meth = (x.method if is_model and isinstance(x, RMAResult) else 'REML') if method_given is None else method_given
        tau2_fix: float | None = None
        if isinstance(x, RMAResult) and x.tau2_fix:
            tau2_fix = x.tau2
        if tau2 is not None:
            tau2_fix = tau2
        tst = (x.test if isinstance(x, RMAResult) else 'z') if test is None else test
        if tst != 'z':
            exact = True
        wtd = (x.weighted if isinstance(x, RMAResult) else True) if weighted is None else bool(weighted)
        try:
            tmp = rma_uni(yi, v, method=meth, tau2=tau2_fix, test=tst, weighted=wtd)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise ValueError('Could not fit random-effects model.') from exc
        vt = 1 / float(np.mean(1 / v))
        est = float(tmp.beta[0])
        tau2_out = tmp.tau2
        pval = float(tmp.pval[0])
        meth_out = meth

        def gen_fn(n: float, target_: float | None, newest: bool = False) -> Any:
            return _fsn_gen(n, yi, v, vt, est, tau2_out, tau2_fix, tst, wtd, target_, alpha, exact, meth, mumiss, maxint, newest, gen)

        def root_of(extend: Literal['upX', 'downX'], target_: float | None) -> float:
            try:
                r = uniroot(lambda n: gen_fn(n, target_), ival[0], ival[1], tol=tol, maxiter=maxiter, extend_int=extend).root
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                raise ValueError('Could not find fail-safe N based on a random-effects model.') from exc
            return rnd_fsn(min(r, maxint))

        got: dict[str, float] | None = None
        if target is None:
            if pval >= alpha:
                fsnum = 0.0
            else:
                fsnum = root_of('upX', None)
                got = gen_fn(fsnum, None, True)
            tgt = math.nan
        else:
            if np.sign(tgt) != np.sign(est):
                tgt = -1 * tgt
            if tgt == 0:
                fsnum = math.inf
            elif abs(tgt) >= abs(est):
                fsnum = 0.0
            else:
                fsnum = root_of('downX' if est > 0 else 'upX', tgt)
                got = gen_fn(fsnum, tgt, True)
        if fsnum == 0:
            est_fsn, tau2_fsn, pval_fsn = est, tau2_out, pval
        elif got is not None:
            est_fsn, tau2_fsn, pval_fsn = got['est_fsn'], got['tau2_fsn'], got['pval_fsn']
        if fsnum >= maxint:
            ub_sign = '>'
    return FsnResult(
        type=typ,
        fsnum=fsnum,
        est=est,
        tau2=tau2_out,
        pval=pval,
        alpha=alpha,
        target=tgt,
        method=meth_out,
        est_fsn=est_fsn,
        tau2_fsn=tau2_fsn,
        pval_fsn=pval_fsn,
        ub_sign=ub_sign,
        digits=d,
    )


def _fsn_gen(
    fsnum: float,
    yi: FloatArray,
    vi: FloatArray,
    vt: float,
    est: float,
    tau2: float,
    tau2_fix: float | None,
    test: str,
    weighted: bool,
    target: float | None,
    alpha: float,
    exact: bool,
    method: str,
    mumiss: float,
    maxint: float,
    newest: bool,
    rng: RRandom,
) -> Any:
    """Port of metafor's internal ``.fsn.gen``: the p-value (or estimate) minus its target after adding ``fsnum`` null studies."""
    n = math.floor(fsnum)
    n = min(n, int(maxint))
    if n < 0:
        raise ValueError('invalid arguments')
    k = yi.shape[0]
    if exact and n <= 5000:
        yinew = np.concatenate([yi, _scale(rng.rnorm(n), n) * math.sqrt(vt + tau2) + mumiss])
        vinew = np.concatenate([vi, np.full(n, vt)])
        tmp = rma_uni(yinew, vinew, method=method, tau2=tau2_fix, test=test, weighted=weighted)
        est_fsn, tau2_fsn, pval_fsn = float(tmp.beta[0]), tmp.tau2, float(tmp.pval[0])
        if target is None and mumiss != 0 and np.sign(est_fsn) == np.sign(mumiss):
            pval_fsn = 1.0
    else:
        est_fsn = math.nan
        if method in FIXED_EFFECTS:
            tau2_fsn = 0.0
        else:
            est_fsn = (k * est + n * mumiss) / (k + n)
            if tau2_fix is None:
                tau2_fsn = ((k - 1) * tau2 + max(0, n - 1) * tau2 + k * (est - est_fsn) ** 2 + n * (mumiss - est_fsn) ** 2) / (k + n - 1)
                if target is None:
                    tau2_fsn = max(0.0, tau2_fsn)
            else:
                tau2_fsn = tau2
        # the standardised file-drawer values average to mumiss exactly, so they enter through their sum alone
        if weighted:
            w_obs = 1 / (vi + tau2_fsn)
            w_new = n / (vt + tau2_fsn)
            est_fsn = (float(np.sum(w_obs * yi)) + w_new * mumiss) / (float(np.sum(w_obs)) + w_new)
            zval = est_fsn / math.sqrt(1 / (float(np.sum(1 / (vi + tau2_fsn))) + n / (vt + tau2_fsn)))
        else:
            est_fsn = (float(np.sum(yi)) + n * mumiss) / (k + n)
            zval = (k + n) * est_fsn / math.sqrt(float(np.sum(vi + tau2_fsn)) + n * (vt + tau2_fsn))
        pval_fsn = float(2 * stats.norm.sf(abs(zval)))
        if target is None and mumiss != 0 and np.sign(est_fsn * mumiss) == 1:
            pval_fsn = 1.0
    if newest:
        return {'est_fsn': est_fsn, 'tau2_fsn': tau2_fsn, 'pval_fsn': pval_fsn}
    if target is None:
        return 0.0 if n == maxint else pval_fsn - alpha
    return 0.0 if n == maxint else est_fsn - target
