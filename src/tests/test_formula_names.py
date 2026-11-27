import json
import pathlib

import numpy
import pandas
import pytest

import metafopy.formula

REFERENCE = json.loads((pathlib.Path(__file__).parent / 'reference' / 'formula_names.json').read_text())
FORMULAS = [name for name in REFERENCE if name != 'meta']

# the data frame of src/tests/r/generate_formula_names.R
DATA = pandas.DataFrame(
    {
        'x': [1.5, 2.25, 3, 4.75, 5.5, 7],
        'z': [0.5, -1, 2, 0.25, -3, 1.75],
        'year': [1980, 1985, 1990, 1995, 2000, 2005],
        'deltype': ['covert', 'general', 'overt', 'covert', 'general', 'overt'],
        'outcome': ['AL', 'PD', 'AL', 'PD', 'AL', 'PD'],
        'arm': ['ctl', 'trt', 'trt', 'ctl', 'trt', 'ctl'],
    }
)


@pytest.mark.parametrize('formula', FORMULAS)
def test_names_and_matrix_follow_r(formula: str) -> None:
    X, names, _ = metafopy.formula.design_matrix(formula, DATA)
    expected = REFERENCE[formula]
    assert list(names) == ['intrcpt' if name == '(Intercept)' else name for name in expected['names']]
    numpy.testing.assert_allclose(X, numpy.array(expected['columns'], dtype=float).T, rtol=1e-15)
