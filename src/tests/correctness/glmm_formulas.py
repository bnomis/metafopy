"""Independent re-derivations of the ``rma_glmm`` likelihoods and fits, coded from the defining publications.

Rule: this module imports ``numpy`` and ``scipy`` only, never ``metafopy`` (``test_coverage.py`` checks it with ``ast``), and each function cites the
equation it codes, so that an expected value never comes from metafopy's or metafor's source.

A ``derived`` case names a function here as ``glmm_formulas.<name>``, the value of an ``expect`` key; it is called with the arguments of the case that
it declares (see ``cases.formula_value``). See ``agents/specs/correctness-glmm/spec.md``.
"""
