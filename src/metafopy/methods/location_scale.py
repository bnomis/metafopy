"""Registers the methods for location-scale models (``rma_uni(..., scale=)``) with the generic functions."""

import metafopy.methods.ls_anova  # noqa: F401  registers anova for RMALSResult
import metafopy.methods.ls_confint  # noqa: F401  registers confint for RMALSResult
import metafopy.methods.ls_permutest  # noqa: F401  registers permutest for RMALSResult
import metafopy.methods.ls_predict  # noqa: F401  registers predict for RMALSResult
import metafopy.methods.ls_profile  # noqa: F401  registers profile for RMALSResult
import metafopy.methods.ls_resid  # noqa: F401  registers blup, ranef and rstandard for RMALSResult
import metafopy.methods.ls_vif  # noqa: F401  registers vif for RMALSResult

__all__: list[str] = []
