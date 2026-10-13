"""Plot functions mirroring metafor's graphics (matplotlib)."""

from metafopy.plot._baujat import BaujatResult, baujat
from metafopy.plot._forest import ForestResult, addpoly, forest
from metafopy.plot._funnel import FunnelResult, funnel
from metafopy.plot._gosh import PlotGoshResult, plot_gosh
from metafopy.plot._infl import PlotInflResult, plot_infl
from metafopy.plot._labbe import LabbeResult, labbe
from metafopy.plot._profile import plot_profile
from metafopy.plot._qqnorm import QQNormResult, qqnorm
from metafopy.plot._radial import RadialResult, galbraith, radial

__all__ = [
    'BaujatResult',
    'ForestResult',
    'FunnelResult',
    'LabbeResult',
    'PlotGoshResult',
    'PlotInflResult',
    'QQNormResult',
    'RadialResult',
    'addpoly',
    'baujat',
    'forest',
    'funnel',
    'galbraith',
    'labbe',
    'plot_gosh',
    'plot_infl',
    'plot_profile',
    'qqnorm',
    'radial',
]
