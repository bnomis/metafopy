from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# draw plots without a display (and make the images of tests/test_plots_image.py reproducible)
matplotlib.use('Agg')

# make tests/reference_data.py importable as a plain module
sys.path.insert(0, str(Path(__file__).parent))
