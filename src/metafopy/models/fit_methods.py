"""Methods shared by the fitted-model result classes: ``logLik``, ``deviance``, ``AIC``, ``BIC``, ``AICc``, ``fitstats``,
``nobs``, ``df_residual``, ``formula``, ``model_matrix``, ``update`` and ``simulate``.

The fit-statistics methods mirror metafor's ``logLik()``, ``AIC()``, ``BIC()`` and ``fitstats()``, including the table of several models; the rest
forward to :mod:`metafopy.methods.accessors` and :mod:`metafopy.methods.simulate`.
"""

import collections.abc
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.errors

__all__ = ['DataFields', 'FitMethods']

# auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
STAT_NAMES = ['logLik', 'deviance', 'AIC', 'BIC', 'AICc']  # auditme: accept[non-alphabetical-literal] metafor's order


def default_names(fits: collections.abc.Sequence[typing.Any], names: collections.abc.Sequence[str] | None) -> list[str]:
    if names is None:
        return [f'model{i + 1}' for i in range(len(fits))]
    if len(names) != len(fits):
        raise metafopy.errors.InputError(f'got {len(names)} names for {len(fits)} models')
    return [str(n) for n in names]


def warn_if_different_data(fits: collections.abc.Sequence[typing.Any]) -> None:
    """Warn as metafor does when the models were not all fitted to the same ``yi``."""
    yis = [getattr(f, 'yi', None) for f in fits]
    if any(y is None for y in yis):
        return
    arrays = [numpy.asarray(y) for y in yis]
    if any(a.shape != arrays[0].shape or not numpy.array_equal(a, arrays[0], equal_nan=True) for a in arrays[1:]):
        warnings.warn('Models not all fitted to the same data.', stacklevel=3)


class DataFields:
    """The data summaries metafor stores in an ``rma.uni`` object: ``k_all``, ``k_eff``, ``p_eff`` and ``ni`` (``fit.data`` holds the data)."""

    __slots__ = ()

    @property
    def k_all(self) -> int:
        """Number of rows supplied, including those dropped for missing values."""
        return int(typing.cast(typing.Any, self).data.k_all)

    @property
    def k_eff(self) -> int:
        """Number of effective observations (the studies used in the fit)."""
        return int(typing.cast(typing.Any, self).k)

    @property
    def p_eff(self) -> int:
        """Number of effective model coefficients."""
        return int(typing.cast(typing.Any, self).p)

    @property
    def ni(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Sample sizes of the studies used in the fit (``None`` when unknown), as metafor's ``ni``."""
        return typing.cast(numpy.typing.NDArray[numpy.float64] | None, typing.cast(typing.Any, self).data.ni)

    @property
    def M(self) -> numpy.typing.NDArray[numpy.float64]:
        """The marginal variance-covariance matrix of the observed effects, ``diag(vi + tau2)``, metafor's ``M``."""
        fit = typing.cast(typing.Any, self)
        return numpy.diag(numpy.asarray(fit.vi, dtype=numpy.float64) + numpy.asarray(fit.tau2, dtype=numpy.float64))


class FitMethods:
    """Mixin of the generic accessors of metafor fits (``nobs()``, ``df.residual()``, ``formula()``, ...)."""

    __slots__ = ()

    @property
    def b(self) -> numpy.typing.NDArray[numpy.float64]:
        """The estimated coefficients, an old alias of ``beta`` (metafor's ``b``)."""
        return typing.cast(numpy.typing.NDArray[numpy.float64], typing.cast(typing.Any, self).beta)

    def use_reml(self, REML: bool | None) -> bool:
        """``REML`` as given, else whether the model was fitted with REML (as in metafor)."""
        return bool(REML) if REML is not None else getattr(self, 'method', None) == 'REML'

    def logLik(self, REML: bool | None = None) -> float:
        """Log-likelihood; REML unless the model was fitted with ML (as in metafor)."""
        return float(typing.cast(typing.Any, self).fit_stats(self.use_reml(REML)).ll)

    def deviance(self, REML: bool | None = None) -> float:
        return float(typing.cast(typing.Any, self).fit_stats(self.use_reml(REML)).dev)

    def AICc(self, REML: bool | None = None) -> float:
        return float(typing.cast(typing.Any, self).fit_stats(self.use_reml(REML)).AICc)

    def AIC(
        self,
        *others: typing.Any,
        k: float = 2,
        correct: bool = False,
        REML: bool | None = None,
        names: collections.abc.Sequence[str] | None = None,
    ) -> float | pandas.DataFrame:
        """AIC (AICc with ``correct=True``) of this fit, as metafor's ``AIC()``.

        With further fits in ``others`` a frame with the columns ``df`` (number of parameters) and ``AIC`` (or ``AICc``), one row per fit
        (labelled by ``names``, default ``model1``, ``model2``, ...); the estimation method of this fit decides between REML and ML for all of them.
        As in metafor, ``k`` is accepted but not used.
        """
        del k
        reml = self.use_reml(REML)
        row = 'AICc' if correct else 'AIC'
        fits = [self, *others]
        values = [float(getattr(typing.cast(typing.Any, f).fit_stats(reml), row)) for f in fits]
        if not others:
            return values[0]
        warn_if_different_data(fits)
        return pandas.DataFrame({'df': [f.parms for f in fits], row: values}, index=default_names(fits, names))

    def BIC(self, *others: typing.Any, REML: bool | None = None, names: collections.abc.Sequence[str] | None = None) -> float | pandas.DataFrame:
        """BIC of this fit, as metafor's ``BIC()``; with further fits in ``others`` a frame with the columns ``df`` and ``BIC`` (see :meth:`AIC`)."""
        reml = self.use_reml(REML)
        fits = [self, *others]
        values = [float(typing.cast(typing.Any, f).fit_stats(reml).BIC) for f in fits]
        if not others:
            return values[0]
        warn_if_different_data(fits)
        return pandas.DataFrame({'df': [f.parms for f in fits], 'BIC': values}, index=default_names(fits, names))

    def fitstats(self, *others: typing.Any, REML: bool | None = None, names: collections.abc.Sequence[str] | None = None) -> pandas.Series | pandas.DataFrame:
        """``logLik``, ``deviance``, ``AIC``, ``BIC`` and ``AICc`` as a Series, as metafor's ``fitstats()``.

        With further fits in ``others`` a frame with one column per fit (labelled by ``names``, default ``model1``, ``model2``, ...).
        """
        reml = self.use_reml(REML)
        fits = [self, *others]
        columns = [list(typing.cast(typing.Any, f).fit_stats(reml).as_tuple()) for f in fits]
        if not others:
            return pandas.Series(columns[0], index=STAT_NAMES, name='REML' if reml else 'ML')
        warn_if_different_data(fits)
        return pandas.DataFrame(dict(zip(default_names(fits, names), columns, strict=True)), index=STAT_NAMES)

    def coef_summary(self, type: typing.Literal['beta', 'alpha', 'delta'] | None = None) -> typing.Any:
        """See :func:`metafopy.methods.coef_summary.coef_summary`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.coef_summary

        return metafopy.methods.coef_summary.coef_summary(typing.cast(typing.Any, self), type)

    def nobs(self, all: bool = False) -> typing.Any:
        """See :func:`metafopy.methods.accessors.nobs`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.accessors

        return metafopy.methods.accessors.nobs(typing.cast(typing.Any, self), all=all)

    def df_residual(self) -> int:
        """See :func:`metafopy.methods.accessors.df_residual`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.accessors

        return metafopy.methods.accessors.df_residual(typing.cast(typing.Any, self))

    def formula(self, type: str = 'mods') -> str | None:
        """See :func:`metafopy.methods.accessors.formula_of`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.accessors

        return metafopy.methods.accessors.formula_of(typing.cast(typing.Any, self), type)

    @property
    def formula_yi(self) -> str | None:
        """The formula given as ``yi`` (``'yi ~ x'``), if any: metafor's ``formula.yi``."""
        return self.formula('yi')

    @property
    def formula_mods(self) -> str | None:
        """The formula of the moderators (``'~ x'``), if any: metafor's ``formula.mods``."""
        return self.formula('mods')

    def model_matrix(self, asdf: bool = False) -> typing.Any:
        """See :func:`metafopy.methods.accessors.model_matrix`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.accessors

        return metafopy.methods.accessors.model_matrix(typing.cast(typing.Any, self), asdf)

    def update(self, formula: str | None = None, evaluate: bool = True, **kwargs: typing.Any) -> typing.Any:
        """See :func:`metafopy.methods.accessors.update`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.accessors

        return metafopy.methods.accessors.update(typing.cast(typing.Any, self), formula, evaluate, **kwargs)

    def simulate(self, nsim: int = 1, seed: int | None = None, olim: typing.Any = None, **kwargs: typing.Any) -> typing.Any:
        """See :func:`metafopy.methods.simulate.simulate`."""
        # auditme: accept[in-function-import] metafopy.methods imports the result classes, so the import is deferred to avoid a cycle
        import metafopy.methods.simulate

        return metafopy.methods.simulate.simulate(typing.cast(typing.Any, self), nsim, seed, olim, **kwargs)
