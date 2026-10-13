"""Helpers for comparing the artists of a metafopy plot with the geometry recorded from R metafor (tests/r/generate_plots.R)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from metafopy.plot._draw import base_fontsize, collect
from refutil import as_list, close, load_phase2

PLOTS = load_phase2('plots')
RTOL = 1e-8


def flat(v: Any) -> list[Any]:
    """A recorded R vector as a list (jsonlite unboxes length-one vectors)."""
    return [] if v is None else as_list(v)


def _fl(v: Any) -> NDArrayF:
    return np.array([np.nan if a is None else a for a in flat(v)], dtype=float)


type NDArrayF = np.ndarray[Any, np.dtype[np.float64]]


def _recycle(values: Sequence[Any], n: int) -> list[Any]:
    if not values:
        return [None] * n
    return [values[i % len(values)] for i in range(n)]


def r_texts(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in rec.get('text', []):
        labels = flat(t.get('labels'))
        if not labels:
            continue
        n = max(len(labels), len(flat(t.get('x'))), len(flat(t.get('y'))))
        xs, ys = _recycle(flat(t.get('x')), n), _recycle(flat(t.get('y')), n)
        pos, labs = _recycle(flat(t.get('pos')), n), _recycle(labels, n)
        cols = _recycle(flat(t.get('col')), n)
        for i in range(n):
            out.append({'x': xs[i], 'y': ys[i], 'label': labs[i], 'pos': pos[i], 'bold': t.get('font') == 2, 'cex': t.get('cex'), 'col': cols[i]})
    return out


def r_points(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in rec.get('points', []):
        xs, ys = flat(t['x']), flat(t['y'])
        n = max(len(xs), len(ys))
        xs, ys = _recycle(xs, n), _recycle(ys, n)
        pch, cex, col = _recycle(flat(t.get('pch')), n), _recycle(flat(t.get('cex')), n), _recycle(flat(t.get('col')), n)
        for i in range(n):
            if xs[i] is None or ys[i] is None:
                continue
            out.append({'x': xs[i], 'y': ys[i], 'pch': pch[i], 'cex': cex[i], 'col': col[i]})
    return out


def r_segments(rec: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for t in rec.get('segments', []):
        a = [flat(t[k]) for k in ('x0', 'y0', 'x1', 'y1')]
        n = max(len(v) for v in a)
        cols = [_recycle(v, n) for v in a]
        out.extend((cols[0][i], cols[1][i], cols[2][i], cols[3][i]) for i in range(n))
    return out


def r_rects(rec: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for t in rec.get('rect', []):
        a = [flat(t[k]) if not isinstance(t[k], dict) else t[k] for k in ('xleft', 'ybottom', 'xright', 'ytop')]
        if any(isinstance(v, dict) for v in a):
            continue
        n = max(len(v) for v in a)
        cols = [_recycle(v, n) for v in a]
        out.extend((cols[0][i], cols[1][i], cols[2][i], cols[3][i]) for i in range(n))
    return out


def r_ablines(rec: dict[str, Any], key: str) -> list[float]:
    out: list[float] = []
    for t in rec.get('abline', []):
        out.extend(flat(t.get(key)))
    return out


def _same_summary(actual: NDArrayF, expected: Any, what: str) -> None:
    if isinstance(expected, dict):
        assert len(actual) == expected['n'], f'{what}: length {len(actual)} != {expected["n"]}'
        close(float(np.nansum(actual)), expected['sum'], f'{what}: sum', rtol=1e-7)
        close(actual[:3], expected['head'], f'{what}: head')
        close(actual[-3:], expected['tail'], f'{what}: tail')
    else:
        close(actual, expected, what)


def check_texts(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str, compare_col: bool = False) -> None:
    exp = r_texts(rec)
    act = collect(ax)['text']
    assert len(act) == len(exp), f'{what}: {len(act)} texts, expected {len(exp)}'
    for i, (a, e) in enumerate(zip(act, exp, strict=True)):
        x, y = a.get_position()
        tag = f'{what}: text {i} {e["label"]!r}'
        close(x, e['x'], tag + ' x')
        close(y, e['y'], tag + ' y')
        assert a.get_text() == e['label'], tag
        assert a.meta['pos'] == (None if e['pos'] is None else int(e['pos'])), tag + ' pos'
        assert a.meta['bold'] == e['bold'], tag + ' bold'
        if e['cex'] is not None:
            close(a.get_fontsize() / base_fontsize(), e['cex'], tag + ' cex')
        if compare_col and e['col'] is not None:
            assert a.meta['col'] == e['col'], tag + ' col'


def check_points(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str, compare_col: bool = False) -> None:
    exp = r_points(rec)
    act = collect(ax)['point']
    assert len(act) == len(exp), f'{what}: {len(act)} points, expected {len(exp)}'
    for i, (a, e) in enumerate(zip(act, exp, strict=True)):
        tag = f'{what}: point {i}'
        close(a.get_xdata()[0], e['x'], tag + ' x')
        close(a.get_ydata()[0], e['y'], tag + ' y')
        if e['pch'] is not None:
            assert a.meta['pch'] == int(e['pch']), tag + ' pch'
        if e['cex'] is not None:
            close(a.meta['cex'], e['cex'], tag + ' cex')
        if compare_col and e['col'] is not None:
            assert a.meta['col'] == e['col'], tag + ' col'


def check_segments(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str) -> None:
    exp = r_segments(rec)
    act = collect(ax)['segment']
    assert len(act) == len(exp), f'{what}: {len(act)} segments, expected {len(exp)}'
    for i, (a, e) in enumerate(zip(act, exp, strict=True)):
        got = [a.get_xdata()[0], a.get_ydata()[0], a.get_xdata()[1], a.get_ydata()[1]]
        close(got, list(e), f'{what}: segment {i}')


def check_rects(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str) -> None:
    exp = r_rects(rec)
    act = collect(ax)['rect']
    assert len(act) == len(exp), f'{what}: {len(act)} rects, expected {len(exp)}'
    for i, (a, e) in enumerate(zip(act, exp, strict=True)):
        xl, yb, xr, yt = a.meta['xy']
        close([xl, yb, xr, yt], list(e), f'{what}: rect {i}')


def check_polygons(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str, compare_col: bool = False) -> None:
    exp = rec.get('polygon', [])
    act = collect(ax)['polygon']
    assert len(act) == len(exp), f'{what}: {len(act)} polygons, expected {len(exp)}'
    for i, (a, e) in enumerate(zip(act, exp, strict=True)):
        xy = a.meta['xy']
        _same_summary(xy[:, 0], e['x'], f'{what}: polygon {i} x')
        _same_summary(xy[:, 1], e['y'], f'{what}: polygon {i} y')
        if compare_col and e.get('col') is not None:
            assert a.meta['col'] == e['col'], f'{what}: polygon {i} col'


def check_lines(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str, gid: str = 'line') -> None:
    exp = rec.get('lines', [])
    act = collect(ax)[gid]
    assert len(act) == len(exp), f'{what}: {len(act)} lines, expected {len(exp)}'
    for i, (a, e) in enumerate(zip(act, exp, strict=True)):
        _same_summary(np.asarray(a.get_xdata(), dtype=float), e['x'], f'{what}: line {i} x')
        _same_summary(np.asarray(a.get_ydata(), dtype=float), e['y'], f'{what}: line {i} y')


def check_ablines(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str) -> None:
    arts = collect(ax)
    close([a.get_ydata()[0] for a in arts['hline']], r_ablines(rec, 'h'), f'{what}: hlines') if r_ablines(rec, 'h') else None
    assert len(arts['hline']) == len(r_ablines(rec, 'h')), f'{what}: number of hlines'
    assert len(arts['vline']) == len(r_ablines(rec, 'v')), f'{what}: number of vlines'
    if arts['vline']:
        close([a.get_xdata()[0] for a in arts['vline']], r_ablines(rec, 'v'), f'{what}: vlines')


def check_abline_ab(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str) -> None:
    exp = [(flat(t['a'])[0], flat(t['b'])[0]) for t in rec.get('abline', []) if t.get('a') is not None]
    act = [(a.meta['a'], a.meta['b']) for a in collect(ax)['abline']]
    assert len(act) == len(exp), f'{what}: {len(act)} ablines, expected {len(exp)}'
    for i, (g, e) in enumerate(zip(act, exp, strict=True)):
        close(list(g), list(e), f'{what}: abline {i}')


def check_axis(ax: Axes, rec: dict[str, Any], what: str, side: int = 1) -> None:
    axes = [a for a in rec.get('axis', []) if a['side'] == side and a.get('at') is not None]
    if not axes:
        return
    last = axes[-1]
    ticks = ax.get_xticks() if side == 1 else ax.get_yticks()
    close(ticks, last['at'], f'{what}: axis {side} at')
    if last.get('labels') is not None:
        labs = [t.get_text() for t in (ax.get_xticklabels() if side == 1 else ax.get_yticklabels())]
        assert labs == [str(v) for v in flat(last['labels'])], f'{what}: axis {side} labels {labs}'


def check_all(ax: Axes | Sequence[Axes], rec: dict[str, Any], what: str, skip: Iterable[str] = (), compare_col: bool = False) -> None:
    """Compare every recorded primitive (except those in ``skip``) with the artists on ``ax``."""
    skip = set(skip)
    if 'text' not in skip:
        check_texts(ax, rec, what, compare_col)
    if 'point' not in skip:
        check_points(ax, rec, what, compare_col)
    if 'segment' not in skip:
        check_segments(ax, rec, what)
    if 'rect' not in skip:
        check_rects(ax, rec, what)
    if 'polygon' not in skip:
        check_polygons(ax, rec, what, compare_col)
    if 'line' not in skip:
        check_lines(ax, rec, what)
    if 'abline' not in skip:
        check_ablines(ax, rec, what)
        check_abline_ab(ax, rec, what)
    if 'axis' not in skip and isinstance(ax, Axes):
        check_axis(ax, rec, what)
        check_axis(ax, rec, what, side=2)
    if 'usr' not in skip and isinstance(ax, Axes):
        usr = rec['usr']
        close(ax.get_xlim(), usr[:2], f'{what}: xlim')
        close(ax.get_ylim(), usr[2:], f'{what}: ylim')


def check_plain(ax: Axes, rec: dict[str, Any], what: str, labels: bool = True, yaxis: bool = True) -> None:
    """Compare the frame of a plain ``plot()``: user coordinates, ticks and tick labels, axis labels and grid lines."""
    usr = rec['usr']
    close(ax.get_xlim(), usr[:2], f'{what}: xlim')
    close(ax.get_ylim(), usr[2:], f'{what}: ylim')
    at = rec['axticks']
    close(ax.get_xticks(), at['x'], f'{what}: x ticks')
    if yaxis:
        close(ax.get_yticks(), at['y'], f'{what}: y ticks')
    if labels:
        assert [t.get_text() for t in ax.get_xticklabels()] == [str(v) for v in flat(at['xl'])], f'{what}: x tick labels'
        if yaxis:
            assert [t.get_text() for t in ax.get_yticklabels()] == [str(v) for v in flat(at['yl'])], f'{what}: y tick labels'
    plots = rec.get('plot', [])
    if plots:
        p = plots[0]
        if isinstance(p.get('xlab'), str) and not p.get('xlab_expr'):
            assert ax.get_xlabel() == p['xlab'], f'{what}: xlab'
        if isinstance(p.get('ylab'), str) and not p.get('ylab_expr'):
            assert ax.get_ylabel() == p['ylab'], f'{what}: ylab'
    grids = rec.get('grid', [])
    arts = collect(ax)['grid']
    if grids:
        n = len(flat(grids[0]['x'])) + len(flat(grids[0]['y']))
        assert len(arts) == n, f'{what}: grid lines'
        close([a.meta['v'] for a in arts if 'v' in a.meta], grids[0]['x'], f'{what}: grid x')
        close([a.meta['h'] for a in arts if 'h' in a.meta], grids[0]['y'], f'{what}: grid y')
    else:
        assert not arts, f'{what}: unexpected grid'
