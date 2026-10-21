"""The showcase notebook of the docs runs from top to bottom."""

import pathlib

import nbclient
import nbformat
import pytest

NOTEBOOK = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'notebooks' / 'showcase.ipynb'


@pytest.mark.slow
def test_showcase_notebook_executes() -> None:
    pytest.importorskip('altair')
    pytest.importorskip('vl_convert')
    nb = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]  # nbformat ships no annotations
    nbclient.NotebookClient(nb, timeout=900, kernel_name='python3').execute()
    charts = [o for cell in nb.cells if cell.cell_type == 'code' for o in cell.outputs if 'text/html' in o.get('data', {}) and 'vega' in o['data']['text/html']]
    assert len(charts) >= 5
