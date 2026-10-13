"""Plot functions mirroring metafor's graphics (matplotlib)."""

from metafopy.plot._baujat import BaujatResult, baujat
from metafopy.plot._forest import ForestResult, addpoly, forest
from metafopy.plot._funnel import FunnelResult, funnel

__all__ = ['BaujatResult', 'ForestResult', 'FunnelResult', 'addpoly', 'baujat', 'forest', 'funnel']
