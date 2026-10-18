# Transformations

Transformation functions, used as `transf=` or `atransf=` in `predict`, `confint` and the plot functions. metafor's
`transf.ilogit` is `metafopy.transf.ilogit`, and `transf.ipft.hm` is `metafopy.transf.ipft_hm`. Each is also a top-level
function with the mechanical translation of the R name, `metafopy.transf_ilogit` and `metafopy.transf_ipft_hm`.
Functions with `_int` and `_mode` suffixes integrate over or find the mode of the distribution of the true effects, and
take `targs={'tau2': ...}`.

::: metafopy.transf
    options:
      heading_level: 3
      show_root_heading: false
      show_source: false
      filters: public

## R's distribution functions

`plogis`, `qlogis`, `pnorm` and `qnorm` are R's, kept in `metafopy.transf` because metafor code passes them as `transf=`.

::: metafopy.transf.plogis
    options:
      heading_level: 3
      show_source: false

::: metafopy.transf.qlogis
    options:
      heading_level: 3
      show_source: false

::: metafopy.transf.pnorm
    options:
      heading_level: 3
      show_source: false

::: metafopy.transf.qnorm
    options:
      heading_level: 3
      show_source: false
