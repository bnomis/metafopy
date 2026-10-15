"""Plot functions mirroring metafor's graphics (matplotlib)."""

from metafopy.plot.baujat import BaujatResult, baujat
from metafopy.plot.cumul import PlotCumulResult, forest_cumul, plot_cumul
from metafopy.plot.forest import ForestResult, addpoly, forest
from metafopy.plot.funnel import FunnelResult, funnel
from metafopy.plot.gosh import PlotGoshResult, plot_gosh
from metafopy.plot.infl import PlotInflResult, plot_infl
from metafopy.plot.labbe import LabbeResult, labbe
from metafopy.plot.llplot import LlplotResult, llplot
from metafopy.plot.permutest import PlotPermutestResult, plot_permutest
from metafopy.plot.profile import plot_profile
from metafopy.plot.qqnorm import QQNormResult, qqnorm
from metafopy.plot.radial import RadialResult, galbraith, radial
from metafopy.plot.regplot import RegplotResult, points_regplot, regplot
from metafopy.plot.rma import PlotRmaResult, plot_rma
from metafopy.plot.selmodel import SelPlotResult, plot_selmodel
from metafopy.plot.vifplot import PlotVifResult, plot_vif

__all__ = [
    'BaujatResult',
    'ForestResult',
    'FunnelResult',
    'LabbeResult',
    'LlplotResult',
    'PlotCumulResult',
    'PlotGoshResult',
    'PlotInflResult',
    'PlotPermutestResult',
    'PlotRmaResult',
    'PlotVifResult',
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
    'plot_permutest',
    'plot_profile',
    'plot_rma',
    'plot_selmodel',
    'plot_vif',
    'points_regplot',
    'qqnorm',
    'radial',
    'regplot',
]
