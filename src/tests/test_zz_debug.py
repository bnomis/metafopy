import sys
import warnings

import numpy

import test_rma_mv_extras as t


def test_debug() -> None:
    if sys.platform != 'darwin':
        return
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = t.ar(mods=t.DELTYPE, optbeta=True)
    h = r.hessian
    info = {
        'names': r.vc_names,
        'p': r.p,
        'sigma2': r.sigma2,
        'beta': r.beta,
        'shape': None if h is None else h.shape,
        'eig': None if h is None else numpy.linalg.eigvalsh(numpy.nan_to_num(h)),
        'diag': None if h is None else numpy.diag(h),
        'warn': [str(x.message) for x in w],
        'll': r.fit_stats,
    }
    raise AssertionError(repr(info))
