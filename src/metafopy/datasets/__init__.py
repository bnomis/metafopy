"""Example datasets from the R ``metadat`` package.

The CSV files under ``data/`` are exported from metadat by
``src/tests/r/export_datasets.R`` and ship with metafopy so documentation and
tests can reproduce metafor's published examples. metadat is licensed
GPL (>= 2); see https://cran.r-project.org/package=metadat for the sources of
each dataset.
"""

from metafopy.datasets.base import list_datasets, load

# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
__all__ = ['list_datasets', 'load']
