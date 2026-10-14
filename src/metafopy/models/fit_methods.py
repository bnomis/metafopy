"""Methods shared by the fitted-model result classes: ``nobs``, ``df_residual``, ``formula``, ``model_matrix``, ``update`` and ``simulate``.

They forward to :mod:`metafopy.methods.accessors` and :mod:`metafopy.methods.simulate`.
"""

import typing

__all__ = ['FitMethods']


class FitMethods:
    """Mixin of the generic accessors of metafor fits (``nobs()``, ``df.residual()``, ``formula()``, ...)."""

    __slots__ = ()

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
