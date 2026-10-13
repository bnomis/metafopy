"""R's default random number generator, so that randomised procedures reproduce metafor's results.

``set.seed(seed)`` followed by ``runif()``/``sample()`` in R is reproduced
exactly: R seeds its Mersenne-Twister (MT19937) by scrambling the seed with a
linear congruential generator, turns the 32-bit outputs into doubles with
``fixup``, and (since R 3.6.0) draws ``sample()`` indices by rejection sampling.
The sampling routines follow R's ``src/main/RNG.c`` and ``src/main/random.c``
(GPL-2 | GPL-3).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import special

__all__ = ['RRandom', 'get_rng', 'set_seed']

_N = 624
_I2_32M1 = 2.328306437080797e-10
_MASK = 0xFFFFFFFF


class RRandom:
    """The state of R's ``.Random.seed`` for the Mersenne-Twister with Rejection sampling."""

    def __init__(self, seed: int) -> None:
        s = int(seed) & _MASK
        for _ in range(50):
            s = (69069 * s + 1) & _MASK
        key = np.empty(_N, dtype=np.uint32)
        s = (69069 * s + 1) & _MASK  # dummy[0] (the position slot), overwritten below
        for j in range(_N):
            s = (69069 * s + 1) & _MASK
            key[j] = s
        self._bg = np.random.MT19937()
        self._bg.state = {'bit_generator': 'MT19937', 'state': {'key': key, 'pos': _N}}

    def unif_rand(self) -> float:
        """R's ``unif_rand()``: a double in the open interval (0, 1)."""
        x = int(self._bg.random_raw()) * 2.3283064365386963e-10
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def runif(self, n: int = 1) -> NDArray[np.float64]:
        return np.array([self.unif_rand() for _ in range(n)], dtype=np.float64)

    def norm_rand(self) -> float:
        """R's ``norm_rand()`` (inversion): a standard normal deviate."""
        big = 134217728.0
        u = self.unif_rand()
        u = int(big * u) + self.unif_rand()
        return float(special.ndtri(u / big))

    def rnorm(self, n: int = 1) -> NDArray[np.float64]:
        return np.array([self.norm_rand() for _ in range(n)], dtype=np.float64)

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = math.floor(self.unif_rand() * 65536)
            v = 65536 * v + v1
            n += 16
        if bits < 64:
            v &= (1 << bits) - 1
        return v

    def unif_index(self, dn: int) -> int:
        """R's ``R_unif_index()`` (rejection sampling): an integer in ``0 .. dn-1``."""
        if dn <= 0:
            return 0
        bits = math.ceil(math.log2(dn))
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_int(self, n: int, size: int | None = None, replace: bool = False, prob: Sequence[float] | NDArray[np.float64] | None = None) -> NDArray[np.intp]:
        """R's ``sample.int(n, size, replace, prob)``; returns 1-based indices as R does."""
        k = n if size is None else size
        if prob is not None:
            p = np.asarray(prob, dtype=np.float64).copy()
            if p.shape[0] != n:
                raise ValueError('incorrect number of probabilities')
            if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
                raise ValueError('invalid probabilities')
            p = p / p.sum()
            if replace or k < 2:
                nc = int(np.sum(n * p > 0.1))
                if nc > 200:
                    return self._walker(n, p, k)
                return self._prob_sample_replace(n, p, k)
            return self._prob_sample_no_replace(n, p, k)
        if replace or k < 2:
            return np.array([self.unif_index(n) + 1 for _ in range(k)], dtype=np.intp)
        if k > n:
            raise ValueError("cannot take a sample larger than the population when 'replace = FALSE'")
        x = list(range(n))
        out = np.empty(k, dtype=np.intp)
        m = n
        for i in range(k):
            j = self.unif_index(m)
            out[i] = x[j] + 1
            m -= 1
            x[j] = x[m]
        return out

    def _prob_sample_replace(self, n: int, p: NDArray[np.float64], nans: int) -> NDArray[np.intp]:
        perm = np.arange(1, n + 1)
        p = p.copy()
        revsort(p, perm)
        np.cumsum(p, out=p)
        out = np.empty(nans, dtype=np.intp)
        for i in range(nans):
            ru = self.unif_rand()
            j = 0
            while j < n - 1:
                if ru <= p[j]:
                    break
                j += 1
            out[i] = perm[j]
        return out

    def _prob_sample_no_replace(self, n: int, p: NDArray[np.float64], nans: int) -> NDArray[np.intp]:
        perm = np.arange(1, n + 1)
        p = p.copy()
        revsort(p, perm)
        out = np.empty(nans, dtype=np.intp)
        total = 1.0
        n1 = n - 1
        for i in range(nans):
            rt = total * self.unif_rand()
            mass = 0.0
            j = 0
            while j < n1:
                mass += p[j]
                if rt <= mass:
                    break
                j += 1
            out[i] = perm[j]
            total -= p[j]
            for kk in range(j, n1):
                p[kk] = p[kk + 1]
                perm[kk] = perm[kk + 1]
            n1 -= 1
        return out

    def _walker(self, n: int, p: NDArray[np.float64], nans: int) -> NDArray[np.intp]:
        q = p * n
        hl = [0] * n
        h = -1
        lo = n
        for i in range(n):
            if q[i] < 1.0:
                h += 1
                hl[h] = i
            else:
                lo -= 1
                hl[lo] = i
        a = [0] * n
        if h >= 0 and lo < n:
            for kk in range(n - 1):
                i = hl[kk]
                j = hl[lo]
                a[i] = j
                q[j] += q[i] - 1
                if q[j] < 1.0:
                    lo += 1
                if lo >= n:
                    break
        q = q + np.arange(n)
        out = np.empty(nans, dtype=np.intp)
        for i in range(nans):
            ru = self.unif_rand() * n
            kk = int(ru)
            out[i] = kk + 1 if ru < q[kk] else a[kk] + 1
        return out


def revsort(a: NDArray[np.float64], ib: NDArray[np.intp]) -> None:
    """R's ``revsort()``: sort ``a`` into descending order by heapsort, carrying ``ib`` along (in place)."""
    n = a.shape[0]
    if n <= 1:
        return
    # one-based indexing as in the C source
    A = [0.0, *a.tolist()]
    B = [0, *ib.tolist()]
    ell = (n >> 1) + 1
    ir = n
    while True:
        if ell > 1:
            ell -= 1
            ra = A[ell]
            ii = B[ell]
        else:
            ra = A[ir]
            ii = B[ir]
            A[ir] = A[1]
            B[ir] = B[1]
            ir -= 1
            if ir == 1:
                A[1] = ra
                B[1] = ii
                break
        i = ell
        j = ell << 1
        while j <= ir:
            if j < ir and A[j] > A[j + 1]:
                j += 1
            if ra > A[j]:
                A[i] = A[j]
                B[i] = B[j]
                i = j
                j += i
            else:
                j = ir + 1
        A[i] = ra
        B[i] = ii
    a[:] = A[1:]
    ib[:] = B[1:]


_GLOBAL: RRandom | None = None


def set_seed(seed: int) -> RRandom:
    """Seed the generator used by :func:`metafopy.permutest` and :func:`metafopy.gosh` (R's ``set.seed()``)."""
    global _GLOBAL
    _GLOBAL = RRandom(seed)
    return _GLOBAL


def get_rng() -> RRandom:
    """The module-level generator, seeded from the clock on first use (as R seeds from time and process id)."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = RRandom(int(time.time() * 1000) % 2**31)
    return _GLOBAL
