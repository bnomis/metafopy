"""Plot functions mirroring metafor's graphics (matplotlib)."""

from metafopy.plot.baujat import BaujatResult, baujat
from metafopy.plot.forest import ForestResult, addpoly, forest
from metafopy.plot.funnel import FunnelResult, funnel
from metafopy.plot.gosh import PlotGoshResult, plot_gosh
from metafopy.plot.infl import PlotInflResult, plot_infl
from metafopy.plot.labbe import LabbeResult, labbe
from metafopy.plot.profile import plot_profile
from metafopy.plot.qqnorm import QQNormResult, qqnorm
from metafopy.plot.radial import RadialResult, galbraith, radial

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
