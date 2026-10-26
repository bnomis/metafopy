# Installation

## Requirements

- Python 3.14.7 or later, below 3.15.
- No R installation. The reference numbers from R are committed as fixtures and are only needed by the test suite.

metafopy depends on NumPy, SciPy, pandas, [formulaic](https://matthewwardrop.github.io/formulaic/) (model formulas such as
`'~ ablat'`) and matplotlib (plots). They are installed with it.

## Install

metafopy is pre-release and is not on PyPI yet. Install it from GitHub with pip or uv:

```bash
pip install "metafopy @ git+https://github.com/bnomis/metafopy"
# or
uv add "metafopy @ git+https://github.com/bnomis/metafopy"
```

To work on metafopy itself, clone the repository and follow the development section of the
[README](https://github.com/bnomis/metafopy#install-development).

## Interactive charts (optional)

`metafopy.interactive` builds interactive forest and funnel charts with [Altair](https://altair-viz.github.io/) (see [Plots](guide/plots.md#interactive-charts)),
and the [showcase notebook](notebooks/showcase.ipynb) links them. Install the `interactive` extra to use them (it adds Altair and [vl-convert](https://github.com/vega/vl-convert), which saves the charts as SVG or PNG):

```bash
pip install "metafopy[interactive] @ git+https://github.com/bnomis/metafopy"
```

Results also render as HTML tables in Jupyter, which needs no extra.

## Check the installation

```python
import metafopy

print(metafopy.__version__)
```

```text
0.0.1
```

Then fit a first model on a bundled dataset:

```python
from metafopy import datasets, rma

dat = datasets.load('dat.hackshaw1998')
print(rma(data=dat))
```

```text
Random-Effects Model (k = 37; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 0.0224 (SE = 0.0178)
tau (square root of estimated tau^2 value):      0.1497
I^2 (total heterogeneity / total variability):   29.57%
H^2 (total variability / sampling variability):  1.42

Test for Heterogeneity:
Q(df = 36) = 47.4979, p-val = 0.0952

Model Results:

estimate      se    zval    pval   ci.lb   ci.ub
  0.2189  0.0494  4.4313  <.0001  0.1221  0.3157  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

If this prints a "Random-Effects Model" summary, the install works. `dat.hackshaw1998` holds the studies of Hackshaw, Law and Wald (1997) on lung cancer and passive smoking,
<https://doi.org/10.1136/bmj.315.7114.980>.

## Plots without a display

metafopy draws with matplotlib. On a server or in CI with no display, select a non-interactive backend before plotting:

```python
import matplotlib

matplotlib.use('Agg')
```
