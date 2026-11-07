"""The R generics the audit runs natively in Python (``src/audit/pycall.py``)."""

import audit.pycall
import metafopy
import metafopy.core.labelled


def test_setitem_keeps_an_escalc_frame() -> None:
    """R's ``x$zz <- 1`` keeps the class ``escalc``, as ``x['zz'] = 1`` does in Python; the copy the audit makes must too (audit cases
    ``matrix/$<-/es_*``, where the copy printed as a plain pandas frame)."""
    dat = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    out = audit.pycall.setitem(dat, 'zz', 1)
    assert isinstance(out, metafopy.core.labelled.EscalcFrame)
    assert out.attrs == dat.attrs
    assert 'zz' not in dat.columns
    assert list(out['zz']) == [1] * len(dat)


def test_data_fixture_keeps_an_escalc_dataset() -> None:
    """``{"$data": "dat.konstantopoulos2011"}`` is metadat's ``escalc`` object in R; the copy the audit loads must print as one (audit cases
    ``matrix/print/ks_dat`` and ``matrix/$<-/ks_dat``, which numbered the rows from 0)."""
    out = audit.pycall.resolve_value({'$data': 'dat.konstantopoulos2011'}, {})
    assert isinstance(out, metafopy.core.labelled.EscalcFrame)
