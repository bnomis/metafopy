# Interactive charts

**Version:** 1.0
**Status:** implemented
**Date:** 2026-10-25

Library functions that build interactive Vega-Lite forest and funnel charts
from a fitted model, so a caller gets the showcase notebook's charts without
copying its cells. Today every Altair chart metafopy draws is built by hand in
`docs/notebooks/showcase.ipynb`; nothing under `src/metafopy/` imports Altair.

The first consumer is LitRev, a systematic-review tool that embeds these
charts in a self-contained HTML report and renders the same charts to SVG for
its markdown and PDF routes. Nothing here is specific to LitRev.

metafor has no counterpart: these are a presentation layer over results the
ported functions already compute, so the R oracle covers every number a chart
shows and no new fixture is generated from R.

## Module location

`src/metafopy/interactive/`, a new package:

| File | Contents |
|------|----------|
| `__init__.py` | `__all__ = ['forest_chart', 'funnel_chart', 'chart_data', 'ChartData']` |
| `data.py` | `chart_data()` and `ChartData`: the tidy per-study frame both charts draw from |
| `forest.py` | `forest_chart()` |
| `funnel.py` | `funnel_chart()` |

The package is not imported by `metafopy/__init__.py`: `import
metafopy.interactive` is the only way in, so `import metafopy` never needs
Altair. The showcase notebook is rewritten to call these functions.

## API

```python
@dataclasses.dataclass(frozen=True)
class ChartData:
    """The frames a chart draws: one row per study, and one row per pooled estimate."""

    studies: pandas.DataFrame  # slab, yi, vi, sei, ci_lb, ci_ub, weight, order, href, group, fill; ci_lb/ci_ub on the transformed scale when transformed
    pooled: pandas.DataFrame  # label, estimate, ci_lb, ci_ub, pi_lb, pi_ub, k, tau2, I2 - one row per fit
    measure: str
    transformed: bool


def chart_data(
    x: metafopy.RMAResult | metafopy.models.mv_result.RMAMVResult,
    *,
    transf: collections.abc.Callable[[numpy.typing.ArrayLike], numpy.ndarray] | None = None,
    slab: collections.abc.Sequence[str] | None = None,
    href: collections.abc.Sequence[str | None] | None = None,
    groups: collections.abc.Sequence[str] | None = None,
    group_fits: collections.abc.Mapping[str, metafopy.RMAResult | metafopy.models.mv_result.RMAMVResult] | None = None,
    level: float | None = None,
) -> ChartData:
    """Return the per-study and pooled frames a forest or funnel chart draws.

    Per-study intervals are yi +/- z(level) * sqrt(vi), as metafor's forest()
    draws them, then passed through transf. Weights are weights_() of x,
    as percentages. The pooled rows come from predict(x, transf=transf) for
    x, then one row per entry of group_fits, labelled by its key, in the
    mapping's order. groups gives each study's group label; it is required
    when group_fits is given and must have one entry per study of x.
    href gives each study a link (any URL, including an in-page '#anchor');
    None leaves that study unlinked.

    Raises:
        metafopy.errors.InputError: slab, href or groups has the wrong length,
            or group_fits is given without groups, or x has moderators
            (a forest of a meta-regression is not a pooled estimate).
    """


def forest_chart(
    x: metafopy.RMAResult | metafopy.models.mv_result.RMAMVResult | ChartData,
    *,
    transf: collections.abc.Callable[[numpy.typing.ArrayLike], numpy.ndarray] | None = None,
    xlab: str | None = None,
    refline: float | None = 0.0,
    order: typing.Literal['obs', 'weight', 'input'] = 'input',
    width: int = 560,
    **data_kwargs: object,
) -> altair.VConcatChart:
    """Return an interactive forest plot: a row per study, then the pooled rows.

    Each study is a rule from ci_lb to ci_ub and a square sized by weight;
    each pooled row is a diamond with its prediction interval as a thin
    rule. Hovering a study shows its estimate, interval and weight;
    clicking a linked study follows its href. When x is a fit, the
    remaining keyword arguments go to chart_data(). refline is on the
    transformed scale; None draws no reference line.
    """


def funnel_chart(
    x: metafopy.RMAResult | metafopy.TrimfillResult | ChartData,
    *,
    yaxis: typing.Literal['sei', 'vi'] = 'sei',
    level: float | None = None,
    href: collections.abc.Sequence[str | None] | None = None,
    slab: collections.abc.Sequence[str] | None = None,
    width: int = 420,
) -> altair.LayerChart:
    """Return an interactive funnel plot of a single-level model.

    The pseudo-confidence region is estimate +/- z(level) * sei, with the
    y axis reversed as metafor's funnel() draws it. A TrimfillResult draws
    its imputed studies (fill) as open circles after the observed ones.
    A ChartData draws its studies on the untransformed yi scale against
    the first pooled row's estimate, with its fill column marking imputed
    studies.

    Raises:
        metafopy.errors.NotSupportedError: x is a multilevel fit, for which
            metafor's funnel() has no single-level pseudo-confidence region.
        metafopy.errors.InputError: x is a ChartData with no pooled row.
    """
```

**A `ChartData` is a public input, not only an output.** A caller that has
stored a fit's numbers - a report rendered later from a results file - builds
one from those numbers and charts them without refitting. Both chart
functions accept one, and `chart_data()` is the reference for the columns:
a hand-built `ChartData` missing a column raises `InputError` naming it.

Callers use the returned chart as Altair intends: `chart.to_dict()` for the
Vega-Lite specification, or `vl_convert.vegalite_to_svg(chart.to_dict())`
for a static image. No function here writes a file.

## Implementation

`chart_data()` does all the arithmetic, and both chart functions only map its
columns to marks, so a caller who wants a different chart starts from the
same numbers. It reads only public fields of the fit (`yi`, `vi`, `slab`,
`weights_()`, `predict()`), so the R oracle that pins those pins the chart.

The interactions are the ones the showcase notebook already uses: a
`selection_point` on `slab` that highlights a study in both charts when they
are concatenated by the caller, a tooltip per mark, and an `href` encoding on
the study marks. The notebook's estimator radio button and leave-one-out
dropdown stay in the notebook: they need many fits precomputed, which is a
caller's choice, not a chart's.

Charts carry no data URL and no external resource: data is inlined with
`altair.data_transformers.disable_max_rows()` scoped to the call, never set
globally. Mark colours are fixed values, not a theme, so a chart renders the
same in any embedding.

A study with a missing `yi` or `vi` is omitted, as `forest()` omits it; the
`slab` and `href` sequences index the studies of `x` before omission.

## Error handling

Domain errors are `metafopy.errors` classes, as everywhere in metafopy:

- `InputError` for mismatched lengths and for a moderator model passed to
  `chart_data()`.
- `NotSupportedError` for a multilevel model passed to `funnel_chart()`.
- A missing Altair raises `ImportError` at `import metafopy.interactive`, with
  a message naming the `interactive` extra
  (`pip install "metafopy[interactive] @ git+https://github.com/bnomis/metafopy"`).

## Tests

`src/tests/test_interactive.py`, beside the other flat test modules,
skipped with `pytest.importorskip('altair')`:

- `chart_data()` on `dat.bcg` with `transf=numpy.exp`: the pooled row equals
  `predict(fit, transf=numpy.exp)` and the weights equal `weights_()`.
- Each chart's `to_dict()` validates (Altair validates against the Vega-Lite
  schema) and inlines its data.
- `href` reaches the specification's study marks; `None` entries do not.
- A multilevel fit (`dat.konstantopoulos2011`, `random='~ 1 | district/school'`)
  draws a forest and is refused by `funnel_chart()`.
- `vl_convert.vegalite_to_svg()` renders each chart without error.
- The showcase notebook test keeps passing after the notebook calls these
  functions.

## Documentation

`docs/guide/plots.md` gains an "Interactive charts" section with a runnable
example, covered by `test_docs_examples.py`; `docs/install.md`'s interactive
section names the new module; the function reference picks the package up
from its docstrings.

## External dependencies

Altair (`altair>=6.3`) and vl-convert (`vl-convert-python>=1.9`), both
already in the `interactive` extra; no new dependency.
