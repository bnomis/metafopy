"""Fail-safe N (metafor's ``fsn``).

Rosenthal (1979), doi:10.1037/0033-2909.86.3.638; Orwin (1983), doi:10.3102/10769986008002157;
Rosenberg (2005), doi:10.1111/j.0014-3820.2005.tb01004.x; Viechtbauer's general approach based on
a random-effects model. See https://wviechtb.github.io/metafor/reference/fsn.html.
"""

import dataclasses
import math
import sys
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.bias.inputs
import metafopy.core.rng
import metafopy.core.verbose
import metafopy.core.zeroin
import metafopy.errors
import metafopy.formatters
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.models.tau2

__all__ = ['FsnResult', 'fsn']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type FsnType = typing.Literal['Rosenthal', 'Orwin', 'Rosenberg', 'General', 'Binomial']

EPS = sys.float_info.epsilon


@dataclasses.dataclass(frozen=True, slots=True)
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
        d = metafopy.mfopt.option_digits(digits, self.digits)
        lines = ['', f'Fail-safe N Calculation Using the {self.type} Approach', '']
        alpha = f'{round(self.alpha, d):.7g}'
        if self.type in ('Rosenthal', 'Binomial'):
            lines += [f'Observed Significance Level: {metafopy.bias.inputs.fp(self.pval, d)}', f'Target Significance Level:   {alpha}']
        elif self.type == 'Orwin':
            lines += [f'Average Effect Size: {metafopy.bias.inputs.fx(self.est, d)}', f'Target Effect Size:  {metafopy.bias.inputs.fx(self.target, d)}']
        elif self.type == 'Rosenberg':
            left = ' ' if self.est < 0 else ''
            lines += [
                f'Average Effect Size:         {metafopy.bias.inputs.fx(self.est, d, left)}',
                f'Observed Significance Level: {left}{metafopy.bias.inputs.fp(self.pval, d)}',
                f'Target Significance Level:   {left}{alpha}',
            ]
        else:
            left = ' ' if self.est < 0 else ''
            right = ' ' if self.est_fsn < 0 else ''
            file_drawer = self.fsnum > 0
            line = f'Average Effect Size:         {metafopy.bias.inputs.fx(self.est, d, left)}'
            if file_drawer:
                line += f' (with file drawer: {metafopy.bias.inputs.fx(self.est_fsn, d, right)})'
            lines.append(line)
            if self.method not in metafopy.models.tau2.FIXED_EFFECTS:
                line = f'Amount of Heterogeneity:     {metafopy.bias.inputs.fx(self.tau2, d, left)}'
                if file_drawer:
                    line += f' (with file drawer: {metafopy.bias.inputs.fx(self.tau2_fsn, d, right)})'
                lines.append(line)
            line = f'Observed Significance Level: {left}{metafopy.bias.inputs.fp(self.pval, d)}'
            if file_drawer:
                line += f' (with file drawer: {right}{metafopy.bias.inputs.fp(self.pval_fsn, d)})'
            lines.append(line)
            if math.isnan(self.target):
                lines.append(f'Target Significance Level:   {left}{alpha}')
            else:
                lines.append(f'Target Effect Size:          {metafopy.bias.inputs.fx(self.target, d, left)}')
        num = 'Inf' if math.isinf(self.fsnum) else f'{int(self.fsnum)}'
        return '\n'.join([*lines, '', f'Fail-safe N: {self.ub_sign}{num}', ''])

    def __str__(self) -> str:
        return self.summary()


def rnd_fsn(x: float) -> float:
    if math.isfinite(x) and abs(x - round(x)) >= EPS**0.5:
        return float(math.ceil(x))
    return float(round(x)) if math.isfinite(x) else x


def scale(x: FloatArray, k: int) -> FloatArray:
    if k == 0:
        return x
    if k == 1:
        return numpy.zeros(1)
    return (x - x.mean()) / x.std(ddof=1)


def fsn_fisher(fsnum: float, pi: FloatArray, alpha: float) -> float:
    n = int(fsnum)
    x2 = -2 * float(numpy.sum(numpy.log(numpy.concatenate([pi, numpy.full(n, 0.5)]))))
    return float(scipy.stats.chi2.sf(x2, 2 * (pi.shape[0] + fsnum))) - alpha


def wmean(x: FloatArray, w: FloatArray) -> float:
    return float(numpy.sum(x * w) / numpy.sum(w))


def fsn(
    x: metafopy.models.rma_uni.RMAResult | numpy.typing.ArrayLike | str,
    vi: numpy.typing.ArrayLike | str | None = None,
    *,
    sei: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    data: pandas.DataFrame | None = None,
    type: FsnType | None = None,
    alpha: float = 0.05,
    target: float | None = None,
    method: str | None = None,
    exact: bool = False,
    verbose: bool | int = False,
    digits: int | None = None,
    pool: typing.Literal['stouffer', 'fisher'] = 'stouffer',
    mumiss: float = 0,
    interval: tuple[float, float] | None = None,
    maxint: float = 1e7,
    tol: float = EPS**0.25,
    maxiter: int = 1000,
    tau2: float | None = None,
    test: str | None = None,
    weighted: bool | None = None,
    seed: int | None = None,
    rng: metafopy.core.rng.RRandom | None = None,
) -> FsnResult:
    """Fail-safe N calculation.

    ``type='General'`` (the only choice for a fitted model) adds studies with an average effect of
    ``mumiss`` until the model's p-value exceeds ``alpha`` (or the estimate reaches ``target``). With
    ``exact=True`` a model is refitted to data in which the file-drawer studies are random normal
    deviates, standardised to mean ``mumiss`` and the expected variance; those come from the R-compatible
    generator (:func:`metafopy.set_seed`, or ``seed`` / ``rng`` here). ``verbose`` passes on to the
    random-effects fit and prints each step of the search for the fail-safe N on stdout, as metafor does.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    gen = rng if rng is not None else (metafopy.core.rng.set_seed(seed) if seed is not None else metafopy.core.rng.get_rng())
    typ: str
    method_given = method
    if metafopy.methods.fitview.is_rma(x):
        metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.FIXED_UNI_ONLY)
        if not isinstance(x, metafopy.models.rma_uni.RMAResult):
            # metafor 5.2-1 fails on rma.mh and rma.peto objects with an internal error ("invalid 'y' type in 'x && y'")
            raise metafopy.errors.NotSupportedError(f'Method not available for objects of class "{metafopy.methods.fitview.r_class(x)[0]}".')
    is_model = isinstance(x, metafopy.models.rma_uni.RMAResult)
    if isinstance(x, metafopy.models.rma_uni.RMAResult):
        if not x.int_only:
            raise metafopy.errors.InputError('Method only applicable to models without moderators.')
        if type is not None and type != 'General':
            warnings.warn("Setting type='General' when using fsn() on a model object.", stacklevel=2)
        typ = 'General'
        if x.weights is not None:
            raise metafopy.errors.InputError('Cannot use function on models with custom weights.')
        yi, v, _ = metafopy.bias.inputs.data_of(x)
    else:
        typ_in = 'Rosenthal' if type is None else type
        options = ['rosenthal', 'binomial', 'orwin', 'rosenberg', 'general']
        match = [o for o in options if o.startswith(typ_in.lower())]
        if not match:
            raise metafopy.errors.InputError("Unknown 'type' specified.")
        typ = match[0].capitalize()
        v_in: typing.Any = vi
        if typ == 'Orwin' and weighted is False:
            v_in, sei = 1.0, None
        elif typ == 'Orwin' and weighted is None and vi is None and sei is None:
            v_in = 1.0
        if isinstance(v_in, float) and v_in == 1.0:
            x_vec = metafopy.bias.inputs.as_vector(x, 'x', data)
            if x_vec is None:
                raise metafopy.errors.InputError("Argument 'x' must be a vector with estimates or an 'rma' model object.")
            ones = numpy.ones(x_vec.shape[0])
            yi, v_opt, _ = metafopy.bias.inputs.vector_inputs(x, ones, None, subset=subset, data=data, what='the calculation')
        else:
            yi, v_opt, _ = metafopy.bias.inputs.vector_inputs(
                x, v_in, sei, subset=subset, data=data, what='the calculation', vi_required=typ in ('Rosenthal', 'Rosenberg', 'General')
            )
        if v_opt is None:
            if typ != 'Binomial':
                raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
            v_opt = numpy.ones_like(yi)
        v = v_opt
    if numpy.any(v <= 0):
        raise metafopy.errors.InputError('Cannot use function when there are non-positive sampling variances in the data.')
    k = yi.shape[0]
    if k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    d = 4 if digits is None else digits
    ival = (0.0, k * 50.0) if interval is None else interval

    est = tau2_out = pval = est_fsn = tau2_fsn = pval_fsn = math.nan
    ub_sign = ''
    meth_out: str | None = None
    tgt = math.nan if target is None else target
    fsnum: float

    if typ == 'Rosenthal' and pool == 'stouffer':
        zi = yi / numpy.sqrt(v)
        z_avg = abs(float(numpy.sum(zi)) / math.sqrt(k))
        pval = float(scipy.stats.norm.sf(z_avg))
        fsnum = rnd_fsn(max(0.0, k * (z_avg / float(scipy.stats.norm.isf(alpha))) ** 2 - k))
        tgt = math.nan
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    elif typ == 'Rosenthal':
        zi = yi / numpy.sqrt(v)
        pi = scipy.stats.norm.sf(numpy.abs(zi))
        pval = fsn_fisher(0, pi, 0.0)
        if pval >= alpha:
            fsnum = 0.0
        else:
            fsnum = metafopy.core.zeroin.uniroot(lambda n: fsn_fisher(n, pi, alpha), ival[0], ival[1], tol=tol, maxiter=maxiter, extend_int='upX').root
        fsnum = rnd_fsn(fsnum)
        tgt = math.nan
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    elif typ == 'Binomial':
        kpos = int(numpy.sum(yi > 0))
        pval = float(scipy.stats.binomtest(kpos, k).pvalue)
        fsnum = 0.0
        if pval < alpha:
            pnew = pval
            while pnew < alpha:
                fsnum += 2
                pnew = float(scipy.stats.binomtest(kpos + int(fsnum / 2), k + int(fsnum)).pvalue)
        fsnum = rnd_fsn(fsnum)
        tgt = math.nan
    elif typ == 'Orwin':
        est = wmean(yi, 1 / v)
        if target is None:
            tgt = est / 2
        if tgt == 0:
            fsnum = math.inf
        else:
            if numpy.sign(tgt) != numpy.sign(est):
                tgt = -1 * tgt
            fsnum = max(0.0, k * (est - tgt) / tgt)
        fsnum = rnd_fsn(fsnum)
    elif typ == 'Rosenberg':
        wi = 1 / v
        est = wmean(yi, wi)
        zval = est / math.sqrt(1 / float(numpy.sum(wi)))
        pval = float(2 * scipy.stats.norm.sf(abs(zval)))
        vt = 1 / float(numpy.mean(1 / v))
        fsnum = rnd_fsn(max(0.0, ((float(numpy.sum(wi * yi)) / float(scipy.stats.norm.isf(alpha / 2))) ** 2 - float(numpy.sum(wi))) * vt))
        tgt = math.nan
    else:
        meth = (x.method if is_model and isinstance(x, metafopy.models.rma_uni.RMAResult) else 'REML') if method_given is None else method_given
        tau2_fix: float | None = None
        if isinstance(x, metafopy.models.rma_uni.RMAResult) and x.tau2_fix:
            tau2_fix = x.tau2
        if tau2 is not None:
            tau2_fix = tau2
        tst = (x.test if isinstance(x, metafopy.models.rma_uni.RMAResult) else 'z') if test is None else test
        if tst != 'z':
            exact = True
        wtd = (x.weighted if isinstance(x, metafopy.models.rma_uni.RMAResult) else True) if weighted is None else bool(weighted)
        try:
            tmp = metafopy.models.rma_uni.rma_uni(yi, v, method=meth, tau2=tau2_fix, test=tst, weighted=wtd, verbose=verbose)
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            raise metafopy.errors.InputError('Could not fit random-effects model.') from exc
        vt = 1 / float(numpy.mean(1 / v))
        est = float(tmp.beta[0])
        tau2_out = tmp.tau2
        pval = float(tmp.pval[0])
        meth_out = meth

        def gen_fn(n: float, target_: float | None, newest: bool = False) -> typing.Any:
            return fsn_gen(
                n,
                yi,
                v,
                vt,
                est,
                tau2_out,
                tau2_fix,
                tst,
                wtd,
                target_,
                alpha,
                exact,
                meth,
                mumiss,
                maxint,
                newest,
                gen,
                verbose=bool(verbose),
                upperint=max(ival),
            )

        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        def root_of(extend: typing.Literal['upX', 'downX'], target_: float | None) -> float:
            try:
                r = metafopy.core.zeroin.uniroot(lambda n: gen_fn(n, target_), ival[0], ival[1], tol=tol, maxiter=maxiter, extend_int=extend).root
            except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
                raise metafopy.errors.InputError('Could not find fail-safe N based on a random-effects model.') from exc
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
            if numpy.sign(tgt) != numpy.sign(est):
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


def fsn_gen(
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
    rng: metafopy.core.rng.RRandom,
    *,
    verbose: bool = False,
    upperint: float = 0.0,
) -> typing.Any:
    """Port of metafor's internal ``.fsn.gen``: the p-value (or estimate) minus its target after adding ``fsnum`` null studies.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    n = math.floor(fsnum)
    n = min(n, int(maxint))
    if n < 0:
        raise metafopy.errors.InputError('invalid arguments')
    k = yi.shape[0]
    if exact and n <= 5000:
        yinew = numpy.concatenate([yi, scale(rng.rnorm(n), n) * math.sqrt(vt + tau2) + mumiss])
        vinew = numpy.concatenate([vi, numpy.full(n, vt)])
        tmp = metafopy.models.rma_uni.rma_uni(yinew, vinew, method=method, tau2=tau2_fix, test=test, weighted=weighted)
        est_fsn, tau2_fsn, pval_fsn = float(tmp.beta[0]), tmp.tau2, float(tmp.pval[0])
        if target is None and mumiss != 0 and numpy.sign(est_fsn) == numpy.sign(mumiss):
            pval_fsn = 1.0
    else:
        est_fsn = math.nan
        if method in metafopy.models.tau2.FIXED_EFFECTS:
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
            est_fsn = (float(numpy.sum(w_obs * yi)) + w_new * mumiss) / (float(numpy.sum(w_obs)) + w_new)
            zval = est_fsn / math.sqrt(1 / (float(numpy.sum(1 / (vi + tau2_fsn))) + n / (vt + tau2_fsn)))
        else:
            est_fsn = (float(numpy.sum(yi)) + n * mumiss) / (k + n)
            zval = (k + n) * est_fsn / math.sqrt(float(numpy.sum(vi + tau2_fsn)) + n * (vt + tau2_fsn))
        pval_fsn = float(2 * scipy.stats.norm.sf(abs(zval)))
        if target is None and mumiss != 0 and numpy.sign(est_fsn * mumiss) == 1:
            pval_fsn = 1.0
    if newest:
        return {'est_fsn': est_fsn, 'tau2_fsn': tau2_fsn, 'pval_fsn': pval_fsn}
    diff = (0.0 if n == maxint else pval_fsn - alpha) if target is None else (0.0 if n == maxint else est_fsn - target)
    if verbose:
        fmtx = metafopy.formatters.fmtx
        width = len(metafopy.formatters.r_as_character(upperint)) + 1
        tail = f'pval = {fmtx(pval_fsn)}  alpha = {fmtx(alpha)}' if target is None else f'target = {fmtx(target)}'
        metafopy.core.verbose.cat(f'fsnum = {n:>{width}d}  est = {fmtx(est_fsn, flag=" ")}  tau2 = {fmtx(tau2_fsn)}  {tail}  diff = {fmtx(diff, flag=" ")} \n')
    return diff
