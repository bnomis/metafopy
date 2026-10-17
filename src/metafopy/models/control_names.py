"""The names metafor reads from a ``control`` list in any of its functions.

metafor merges ``control`` into the settings of the function with ``pmatch`` and ignores what does not match, so one ``control`` list can be passed
to ``rma()``, ``confint()``, ``permutest()`` and the other functions, each taking the entries it uses (see
`rma.uni.r <https://github.com/wviechtb/metafor/blob/master/R/rma.uni.r>`_, ``con.pos <- pmatch(names(control), names(con))``).
metafopy raises for an entry no function of metafor reads, so that a misspelt setting is not lost, and ignores the entries that another function
reads (see ``docs/differences.md``).
"""

import collections.abc

R_CONTROL_NAMES = frozenset(
    {
        'REMLf',
        'abstol',
        'alpha_init',
        'alpha_max',
        'alpha_min',
        'alternative',
        'beta_init',
        'check_k_gtr_1',
        'cholesky',
        'cialt',
        'cl',
        'clogisticCtrl',
        'clogitCtrl',
        'comptol',
        'delta_init',
        'delta_max',
        'delta_min',
        'deltatol',
        'distfac',
        'dnchgcalc',
        'dnchgprec',
        'eptries',
        'evtol',
        'extendInt',
        'gamma2',
        'gamma2_init',
        'glmCtrl',
        'glmerCtrl',
        'hes_alpha_fix',
        'hes_beta_fix',
        'hes_delta_fix',
        'hes_omega2_fix',
        'hes_tau2_fix',
        'hessianCtrl',
        'hesspack',
        'hesstol',
        'htransf',
        'intCtrl',
        'll0check',
        'maxit',
        'maxiter',
        'mfmaxit',
        'ncpus',
        'ndeps',
        'nearpd',
        'omega2_init',
        'omega2tol',
        'optCtrl',
        'optimizer',
        'optmethod',
        'p2defn',
        'package',
        'parallel',
        'phi',
        'phi_init',
        'pval_min',
        'reltol',
        'rho',
        'rho_init',
        'scaleX',
        'scaleZ',
        'sigma2',
        'sigma2_init',
        'stat',
        'stepadj',
        'tau2',
        'tau2_init',
        'tau2_max',
        'tau2_min',
        'tau2i_init',
        'tau2tol',
        'threshold',
        'tol',
        'vc_max',
        'vc_min',
        'verbose',
    }
)


def read_elsewhere(key: str) -> bool:
    """Whether some function of metafor reads the ``control`` entry ``key`` (written with ``.`` or ``_``), so that this one ignores it as R does."""
    return key.replace('.', '_') in R_CONTROL_NAMES


def ignore_foreign(control: collections.abc.Mapping[str, object], known: collections.abc.Collection[str]) -> dict[str, object]:
    """The entries of ``control`` the function reads, with the entries only other functions of metafor read dropped.

    Entries that no function of metafor reads stay in the result, for the caller to refuse.
    """
    return {key: value for key, value in control.items() if key.replace('.', '_') in known or not read_elsewhere(key)}
