"""R's ``dbinom()`` for integer arguments, bit for bit (Loader's saddle-point algorithm as R 4.5.1 implements it).

``gosh()`` draws the sizes of the random subsets with ``sample(prob = dbinom(...))``. The binomial probabilities of the sizes ``m`` and
``k - m`` are equal mathematically, but not in the last bit of a double, and R's sampler orders the probabilities, so a different last bit
gives different subsets for the same seed. SciPy's ``binom.pmf`` rounds differently, hence this port of ``dbinom_raw()``, ``bd0()`` and
``stirlerr()`` from R's ``src/nmath`` (``dbinom.c``, ``bd0.c``, ``stirlerr.c``; GPL >= 2), see
https://github.com/wch/r-source/tree/trunk/src/nmath. Only integer ``x`` and ``n`` are supported, which is all ``dbinom`` is used for here.

Loader (2000), Fast and accurate computation of binomial probabilities, https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf
"""

import math
import sys

import numpy
import numpy.typing

import metafopy.errors

__all__ = ['dbinom', 'dbinom_raw']

FloatArray = numpy.typing.NDArray[numpy.float64]

M_LN_2PI = 1.837877066409345483560659472811
DBL_MIN = sys.float_info.min

# 1/12, 1/360, 1/1260, 1/1680, 1/1188, 691/360360 and 1/156: the coefficients of the Stirling series (stirlerr.c)
S0 = 0.083333333333333333333
S1 = 0.00277777777777777777778
S2 = 0.00079365079365079365079365
S3 = 0.000595238095238095238095238
S4 = 0.0008417508417508417508417508
S5 = 0.0019175269175269175269175262
S6 = 0.0064102564102564102564102561

# stirlerr(n) for n = 0, 0.5, 1, ..., 15 (index 2n; the entry for 0 is a placeholder), as stored in stirlerr.c
SFERR_HALVES = (
    0.0,
    0.1534264097200273452913848,
    0.0810614667953272582196702,
    0.0548141210519176538961390,
    0.0413406959554092940938221,
    0.03316287351993628748511048,
    0.02767792568499833914878929,
    0.02374616365629749597132920,
    0.02079067210376509311152277,
    0.01848845053267318523077934,
    0.01664469118982119216319487,
    0.01513497322191737887351255,
    0.01387612882307074799874573,
    0.01281046524292022692424986,
    0.01189670994589177009505572,
    0.01110455975820691732662991,
    0.010411265261972096497478567,
    0.009799416126158803298389475,
    0.009255462182712732917728637,
    0.008768700134139385462952823,
    0.008330563433362871256469318,
    0.007934114564314020547248100,
    0.007573675487951840794972024,
    0.007244554301320383179543912,
    0.006942840107209529865664152,
    0.006665247032707682442354394,
    0.006408994188004207068439631,
    0.006171712263039457647532867,
    0.005951370112758847735624416,
    0.005746216513010115682023589,
    0.005554733551962801371038690,
)


def stirlerr(n: int) -> float:
    """``log(n!) - log(sqrt(2 pi n) (n/e)^n)``, the error of Stirling's formula, for a non-negative integer ``n``."""
    if n <= 15:
        return SFERR_HALVES[2 * n]
    nn = float(n) * float(n)
    x = float(n)
    if n <= 23:
        return (S0 - (S1 - (S2 - (S3 - (S4 - (S5 - S6 / nn) / nn) / nn) / nn) / nn) / nn) / x
    if n <= 27:
        return (S0 - (S1 - (S2 - (S3 - (S4 - S5 / nn) / nn) / nn) / nn) / nn) / x
    if n <= 86:
        return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / x
    if n <= 205:
        return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / x
    if n <= 6180:
        return (S0 - (S1 - S2 / nn) / nn) / x
    if n <= 15_700_000:
        return (S0 - S1 / nn) / x
    return S0 / x


def bd0(x: float, np: float) -> float:
    """Loader's deviance term ``x log(x/np) + np - x``, accurate also when ``x`` is close to ``np``."""
    if abs(x - np) < 0.1 * (x + np):
        d = x - np
        v = d / (x + np)
        s = math.ldexp(d, -1) * v
        if abs(math.ldexp(s, 1)) < DBL_MIN:
            return math.ldexp(s, 1)
        ej = x * v
        v *= v
        for j in range(1, 1000):
            ej *= v
            s_ = s
            s += ej / ((j << 1) + 1)
            if s == s_:
                return math.ldexp(s, 1)
    lg_x_n = math.log(x / np)
    return x * (lg_x_n - 1.0) + np if x > np else x * lg_x_n + np - x


def pow1p(x: float, y: float) -> float:
    """``(1 + x)^y`` for a non-negative integer ``y``, more accurately than ``pow(1 + x, y)`` for small ``x``."""
    if y <= 4.0:
        if y == 0:
            return 1.0
        if y == 1:
            return x + 1.0
        if y == 2:
            return x * (x + 2.0) + 1.0
        if y == 3:
            return x * (x * (x + 3.0) + 3.0) + 1.0
        return x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0
    xp1 = x + 1.0
    x_ = xp1 - 1.0
    if x_ == x or abs(x) > 0.5:
        return math.pow(xp1, y)
    return math.exp(y * math.log1p(x))


def dbinom_raw(x: int, n: int, p: float, q: float) -> float:
    """The binomial probability of ``x`` successes in ``n`` trials with success probability ``p`` and ``q = 1 - p``."""
    if p == 0:
        return 1.0 if x == 0 else 0.0
    if q == 0:
        return 1.0 if x == n else 0.0
    if x == 0:
        if n == 0:
            return 1.0
        return math.pow(q, n) if p > q else pow1p(-p, float(n))
    if x == n:
        return pow1p(-q, float(n)) if p > q else math.pow(p, n)
    if x < 0 or x > n:
        return 0.0
    lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(float(x), n * p) - bd0(float(n - x), n * q)
    lf = M_LN_2PI + math.log(x) + math.log1p(-x / n)
    return math.exp(lc - 0.5 * lf)


def dbinom(x: numpy.typing.ArrayLike, n: int, p: float) -> FloatArray:
    """``dbinom(x, n, p)`` of R for integer ``x`` (a vector) and ``n``.

    Raises:
        metafopy.errors.InputError: ``n`` is not a non-negative integer, ``p`` is not a probability or an ``x`` is not an integer.
    """
    if n < 0 or int(n) != n or not 0.0 <= p <= 1.0:
        raise metafopy.errors.InputError('dbinom() needs an integer n >= 0 and a probability p')
    xs = numpy.atleast_1d(numpy.asarray(x, dtype=numpy.float64))
    if numpy.any(xs != numpy.round(xs)):
        raise metafopy.errors.InputError('dbinom() needs integer values of x')
    return numpy.array([dbinom_raw(int(v), int(n), p, 1.0 - p) if v >= 0 else 0.0 for v in xs], dtype=numpy.float64)
