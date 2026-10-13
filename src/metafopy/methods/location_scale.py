"""Registers the methods for location-scale models (``rma_uni(..., scale=)``) with the generic functions."""

import metafopy.methods.ls_anova
import metafopy.methods.ls_confint
import metafopy.methods.ls_permutest
import metafopy.methods.ls_predict
import metafopy.methods.ls_profile
import metafopy.methods.ls_resid
import metafopy.methods.ls_vif  # noqa: F401  registers vif for RMALSResult
