"""The studentized range distribution, ported from R's ``ptukey`` and ``qtukey``.

emmeans (Tukey and Dunnett adjustments) calls R's ``ptukey``/``qtukey``. ``qtukey`` is a secant search that stops once
successive iterates differ by less than ``1e-4``, so its values differ from the exact quantile in the fifth digit;
porting R's routines reproduces them where SciPy's ``studentized_range`` would not.

Source: R ``src/nmath/ptukey.c`` and ``src/nmath/qtukey.c`` (GPL-2 | GPL-3), after Copenhaver and Holland (1988),
https://doi.org/10.1080/00949658808811068.
"""

import math

import scipy.special

__all__ = ['ptukey', 'qtukey']

WPROB_C1 = -30.0
WPROB_C2 = -50.0
WPROB_C3 = 60.0
WPROB_BB = 8.0
WPROB_WLAR = 3.0
M_1_SQRT_2PI = 0.398942280401432677939946059934
XLEG = (
    0.981560634246719250690549090149,
    0.904117256370474856678465866119,
    0.769902674194304687036893833213,
    0.587317954286617447296702418941,
    0.367831498998180193752691536644,
    0.125233408511468915472441369464,
)
ALEG = (
    0.047175336386511827194615961485,
    0.106939325995318430960254718194,
    0.160078328543346226334652529543,
    0.203167426723065921749064455810,
    0.233492536538354808760849898925,
    0.249147045813402785000562436043,
)
XLEGQ = (
    0.989400934991649932596154173450,
    0.944575023073232576077988415535,
    0.865631202387831743880467897712,
    0.755404408355003033895101194847,
    0.617876244402643748446671764049,
    0.458016777657227386342419442984,
    0.281603550779258913230460501460,
    0.950125098376374401853193354250e-1,
)
ALEGQ = (
    0.271524594117540948517805724560e-1,
    0.622535239386478928628438369944e-1,
    0.951585116824927848099251076022e-1,
    0.124628971255533872052476282192,
    0.149595988816576732081501730547,
    0.169156519395002538189312079030,
    0.182603415044923588866763667969,
    0.189450610455068496285396723208,
)


def pnorm(x: float, mean: float = 0.0) -> float:
    return float(scipy.special.ndtr(x - mean))


def wprob(w: float, rr: float, cc: float) -> float:
    """Probability integral of Hartley's form of the range."""
    qsqz = w * 0.5
    if qsqz >= WPROB_BB:
        return 1.0
    pr_w = 2 * pnorm(qsqz) - 1.0
    pr_w = pr_w**cc if pr_w >= math.exp(WPROB_C2 / cc) else 0.0
    wincr = 2 if w > WPROB_WLAR else 3
    blb = qsqz
    binc = (WPROB_BB - qsqz) / wincr
    bub = blb + binc
    einsum = 0.0
    cc1 = cc - 1.0
    for _ in range(wincr):
        elsum = 0.0
        a = 0.5 * (bub + blb)
        b = 0.5 * (bub - blb)
        for jj in range(1, 13):
            if jj > 6:
                j = 12 - jj + 1
                xx = XLEG[j - 1]
            else:
                j = jj
                xx = -XLEG[j - 1]
            ac = a + b * xx
            qexpo = ac * ac
            if qexpo > WPROB_C3:
                break
            pplus = 2 * pnorm(ac)
            pminus = 2 * pnorm(ac, w)
            rinsum = pplus * 0.5 - pminus * 0.5
            if rinsum >= math.exp(WPROB_C1 / cc1):
                elsum += (ALEG[j - 1] * math.exp(-(0.5 * qexpo))) * rinsum**cc1
        elsum *= ((2.0 * b) * cc) * M_1_SQRT_2PI
        einsum += elsum
        blb = bub
        bub += binc
    pr_w += einsum
    if pr_w <= math.exp(WPROB_C1 / rr):
        return 0.0
    pr_w = pr_w**rr
    return 1.0 if pr_w >= 1.0 else pr_w


def ptukey(q: float, nmeans: float, df: float, nranges: float = 1.0, lower_tail: bool = True) -> float:
    """Distribution function of the studentized range (R's ``ptukey(q, nmeans, df, nranges)``).

    ``df = inf`` is the range of ``nmeans`` standard normals.
    """
    if math.isnan(q) or math.isnan(nmeans) or math.isnan(df) or math.isnan(nranges):
        return math.nan
    if q <= 0:
        return 0.0 if lower_tail else 1.0
    if df < 2 or nranges < 1 or nmeans < 2:
        return math.nan
    if not math.isfinite(q):
        return 1.0 if lower_tail else 0.0
    if df > 25000.0:
        p = wprob(q, nranges, nmeans)
        return p if lower_tail else 0.5 - p + 0.5
    f2 = df * 0.5
    f2lf = ((f2 * math.log(df)) - (df * math.log(2.0))) - math.lgamma(f2)
    f21 = f2 - 1.0
    ff4 = df * 0.25
    if df <= 100.0:
        ulen = 1.0
    elif df <= 800.0:
        ulen = 0.5
    elif df <= 5000.0:
        ulen = 0.25
    else:
        ulen = 0.125
    f2lf += math.log(ulen)
    ans = 0.0
    otsum = 0.0
    for i in range(1, 51):
        otsum = 0.0
        twa1 = (2 * i - 1) * ulen
        for jj in range(1, 17):
            if jj > 8:
                j = jj - 8 - 1
                t1 = (f2lf + (f21 * math.log(twa1 + (XLEGQ[j] * ulen)))) - (((XLEGQ[j] * ulen) + twa1) * ff4)
            else:
                j = jj - 1
                t1 = (f2lf + (f21 * math.log(twa1 - (XLEGQ[j] * ulen)))) + (((XLEGQ[j] * ulen) - twa1) * ff4)
            if t1 >= -30.0:
                if jj > 8:
                    qsqz = q * math.sqrt(((XLEGQ[j] * ulen) + twa1) * 0.5)
                else:
                    qsqz = q * math.sqrt(((-(XLEGQ[j] * ulen)) + twa1) * 0.5)
                otsum += (wprob(qsqz, nranges, nmeans) * ALEGQ[j]) * math.exp(t1)
        if i * ulen >= 1.0 and otsum <= 1.0e-14:
            break
        ans += otsum
    ans = min(ans, 1.0)
    return ans if lower_tail else 0.5 - ans + 0.5


def qinv(p: float, c: float, v: float) -> float:
    """Initial estimate for the secant search (Odeh and Evans, algorithm AS 70)."""
    ps = 0.5 - 0.5 * p
    yi = math.sqrt(math.log(1.0 / (ps * ps)))
    t = yi + ((((yi * -0.453642210148e-04 + -0.204231210125) * yi + -0.342242088547) * yi + -1.0) * yi + 0.322232421088) / (
        (((yi * 0.38560700634e-02 + 0.103537752850) * yi + 0.531103462366) * yi + 0.588581570495) * yi + 0.993484626060e-01
    )
    if v < 120.0:
        t += (t * t * t + t) / v / 4.0
    q = 0.8832 - 0.2368 * t
    if v < 120.0:
        q += -1.214 / v + 1.208 * t / v
    return t * (q * math.log(c - 1.0) + 1.4142)


def qtukey(p: float, nmeans: float, df: float, nranges: float = 1.0) -> float:
    """Quantile function of the studentized range (R's ``qtukey(p, nmeans, df, nranges)``, lower tail)."""
    if math.isnan(p) or math.isnan(nmeans) or math.isnan(df) or math.isnan(nranges):
        return math.nan
    if df < 2 or nranges < 1 or nmeans < 2 or p < 0 or p > 1:
        return math.nan
    if p == 0:
        return 0.0
    if p == 1:
        return math.inf
    x0 = qinv(p, nmeans, df)
    valx0 = ptukey(x0, nmeans, df, nranges) - p
    x1 = max(0.0, x0 - 1.0) if valx0 > 0.0 else x0 + 1.0
    valx1 = ptukey(x1, nmeans, df, nranges) - p
    ans = 0.0
    for _ in range(1, 50):
        ans = x1 - ((valx1 * (x1 - x0)) / (valx1 - valx0))
        valx0 = valx1
        x0 = x1
        if ans < 0.0:
            ans = 0.0
        valx1 = ptukey(ans, nmeans, df, nranges) - p
        x1 = ans
        if abs(x1 - x0) < 0.0001:
            return ans
    return ans
