# Transformations

Transformation functions, used as `transf=` or `atransf=` in `predict`, `confint` and the plot functions. The module is
`metafopy.transf`. Functions with `_int` and `_mode` suffixes integrate over or find the mode of the distribution of the
true effects, and take `targs={'tau2': ...}`.

::: metafopy.transf
    options:
      heading_level: 3
      show_root_heading: false
      show_source: false
      filters:
        - "^transf_"
        - "^plogis$"
        - "^qlogis$"
        - "^pnorm$"
        - "^qnorm$"
