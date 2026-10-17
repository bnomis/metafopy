import collections.abc
import pathlib
import sys

# make src/tests/reference_data.py importable as a plain module
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib
import pytest

import metafopy.printing

# draw plots without a display (and make the images of test_plots_image.py reproducible)
matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def wide_console() -> collections.abc.Iterator[None]:
    """The printouts stored from R were made at ``width = 200``; test_audit8_fields.py checks the wrapping at 80."""
    metafopy.printing.SETTINGS['width'] = 200
    yield
    metafopy.printing.SETTINGS['width'] = 80
