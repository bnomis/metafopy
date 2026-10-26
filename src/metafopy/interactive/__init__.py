"""Interactive Vega-Lite forest and funnel charts of fitted models, built with Altair.

Import this package explicitly (``import metafopy.interactive``); ``import metafopy`` never loads Altair.
"""

try:
    import altair  # noqa: F401
except ImportError as exc:
    raise ImportError(
        'metafopy.interactive needs Altair; install the extra: pip install "metafopy[interactive] @ git+https://github.com/bnomis/metafopy"'
    ) from exc

from metafopy.interactive.data import ChartData, chart_data
from metafopy.interactive.forest import forest_chart
from metafopy.interactive.funnel import funnel_chart

__all__ = ['ChartData', 'chart_data', 'forest_chart', 'funnel_chart']
