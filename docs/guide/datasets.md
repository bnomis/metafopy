# Datasets

metafopy ships the datasets of the R package [metadat](https://cran.r-project.org/package=metadat) that the metafor examples
use. Load one by its R name, with or without the `dat.` prefix, as a `pandas.DataFrame`:

```python
from metafopy import datasets

print(datasets.list_datasets()[:5])
dat = datasets.load('dat.bcg')
print(dat.head())
print(datasets.load('bcg').equals(dat))
```

```text
['dat.assink2016', 'dat.bangertdrowns2004', 'dat.bcg', 'dat.berkey1998', 'dat.cohen1981']
   trial                author  year  tpos  ...  cpos   cneg  ablat      alloc
0      1               Aronson  1948     4  ...    11    128     44     random
1      2      Ferguson & Simes  1949     6  ...    29    274     55     random
2      3       Rosenthal et al  1960     3  ...    11    209     42     random
3      4     Hart & Sutherland  1977    62  ...   248  12619     52     random
4      5  Frimodt-Moller et al  1973    33  ...    47   5761     13  alternate

[5 rows x 9 columns]
True
```

`list_datasets()` lists every bundled dataset. Each `load` returns a fresh copy that you can modify.

The metadat documentation describes the variables and cites the original source of each dataset:
<https://wviechtb.github.io/metadat/>. Columns keep their R names, including dots (`or.lb`), so use brackets
(`dat['or.lb']`) to access them.

## Trees

`dat.moura2021` comes with a phylogenetic tree, loaded with `datasets.load_tree('dat.moura2021.tree')`. The result is a
`PhyloTree` with the tip labels (`tip_labels`) and the variance-covariance or correlation matrix of the tips (`vcv`), which is
what `rma_mv` needs for phylogenetic models. See [Multilevel and multivariate models](multilevel.md#known-correlation-matrices).

## Using your own data

Any `DataFrame` works. Give column names as strings with `data=`, or pass arrays directly. Missing values are handled as in
metafor: `na_action='omit'` (the default) drops studies with missing values, `'fail'` raises an error.
