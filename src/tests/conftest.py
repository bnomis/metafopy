import pathlib
import sys

# make src/tests/reference_data.py importable as a plain module
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib

# draw plots without a display (and make the images of test_plots_image.py reproducible)
matplotlib.use('Agg')
