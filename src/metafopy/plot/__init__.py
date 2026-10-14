"""Plot functions mirroring metafor's graphics (matplotlib)."""

from metafopy.plot.baujat import BaujatResult, baujat
from metafopy.plot.cumul import PlotCumulResult, forest_cumul, plot_cumul
from metafopy.plot.forest import ForestResult, addpoly, forest
from metafopy.plot.funnel import FunnelResult, funnel
from metafopy.plot.gosh import PlotGoshResult, plot_gosh
from metafopy.plot.infl import PlotInflResult, plot_infl
from metafopy.plot.labbe import LabbeResult, labbe
from metafopy.plot.llplot import LlplotResult, llplot
from metafopy.plot.profile import plot_profile
from metafopy.plot.qqnorm import QQNormResult, qqnorm
from metafopy.plot.radial import RadialResult, galbraith, radial
from metafopy.plot.regplot import RegplotResult, regplot
from metafopy.plot.selmodel import SelPlotResult, plot_selmodel

__all__ = [
    'BaujatResult',
    'ForestResult',
    'FunnelResult',
    'LabbeResult',
    'LlplotResult',
    'PlotCumulResult',
    'PlotGoshResult',
    'PlotInflResult',
    'QQNormResult',
    'RadialResult',
    'RegplotResult',
    'SelPlotResult',
    'addpoly',
    'baujat',
    'forest',
    'forest_cumul',
    'funnel',
    'galbraith',
    'labbe',
    'llplot',
    'plot_cumul',
    'plot_gosh',
    'plot_infl',
    'plot_profile',
    'plot_selmodel',
    'qqnorm',
    'radial',
    'regplot',
]
