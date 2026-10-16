"""Paths and tolerances of the audit harness (the one place both are decided)."""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = pathlib.Path(__file__).resolve().parent
METAFOPY_SRC = pathlib.Path(os.environ.get('METAFOPY_AUDIT_SRC', ROOT / 'src'))
R_DIR = PACKAGE / 'r'
DATA_DIR = PACKAGE / 'data'
WORK = pathlib.Path(os.environ.get('METAFOPY_AUDIT_WORK', ROOT / '.audit'))
R_LIB = pathlib.Path(os.environ.get('METAFOPY_AUDIT_RLIB', WORK / 'Rlib'))

METAFOR_VERSION = '5.2-1'

# Numbers closer than RTOL (relative) or ATOL (absolute) are equal. Between RTOL and DRIFT_RTOL a difference is recorded as drift (counted, not
# listed as wrong): optimiser stopping rules differ by that much between R and SciPy. Beyond DRIFT_RTOL it is a value difference.
RTOL = 1e-6
ATOL = 1e-8
DRIFT_RTOL = 1e-4

CASE_TIMEOUT = 120.0
SEED = 1234
MEMORY_LIMIT = 4 * 1024**3
