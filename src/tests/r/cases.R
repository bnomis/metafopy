# Case definitions shared by the fixture scripts.
#
# Each case names a metadat dataset, how to compute effect sizes from it
# (an escalc() argument list, or NULL when the data already carry yi/vi), and
# an optional moderator formula used for meta-regression fixtures.

cases <- list(
  bcg_rr          = list(data = "dat.bcg", escalc = list(measure = "RR", ai = "tpos", bi = "tneg", ci = "cpos", di = "cneg"), mods = "~ ablat"),
  bcg_or          = list(data = "dat.bcg", escalc = list(measure = "OR", ai = "tpos", bi = "tneg", ci = "cpos", di = "cneg"), mods = "~ ablat + year"),
  bcg_rd          = list(data = "dat.bcg", escalc = list(measure = "RD", ai = "tpos", bi = "tneg", ci = "cpos", di = "cneg"), mods = NULL),
  colditz1994_rr  = list(data = "dat.colditz1994", escalc = list(measure = "RR", ai = "tpos", bi = "tneg", ci = "cpos", di = "cneg"), mods = "~ alloc"),
  normand1999_smd = list(data = "dat.normand1999", escalc = list(measure = "SMD", m1i = "m1i", sd1i = "sd1i", n1i = "n1i", m2i = "m2i", sd2i = "sd2i", n2i = "n2i"), mods = NULL),
  normand1999_md  = list(data = "dat.normand1999", escalc = list(measure = "MD", m1i = "m1i", sd1i = "sd1i", n1i = "n1i", m2i = "m2i", sd2i = "sd2i", n2i = "n2i"), mods = NULL),
  gibson2002_smd  = list(data = "dat.gibson2002", escalc = list(measure = "SMD", m1i = "m1i", sd1i = "sd1i", n1i = "n1i", m2i = "m2i", sd2i = "sd2i", n2i = "n2i"), mods = NULL),
  curtis1998_rom  = list(data = "dat.curtis1998", escalc = list(measure = "ROM", m1i = "m1i", sd1i = "sd1i", n1i = "n1i", m2i = "m2i", sd2i = "sd2i", n2i = "n2i"), mods = "~ fungrp"),
  hackshaw1998    = list(data = "dat.hackshaw1998", escalc = NULL, mods = "~ design"),
  raudenbush1985  = list(data = "dat.raudenbush1985", escalc = NULL, mods = "~ weeks"),
  bangertdrowns2004 = list(data = "dat.bangertdrowns2004", escalc = NULL, mods = "~ length"),
  konstantopoulos2011 = list(data = "dat.konstantopoulos2011", escalc = NULL, mods = "~ year"),
  assink2016      = list(data = "dat.assink2016", escalc = NULL, mods = "~ pubstatus"),
  molloy2014_zcor = list(data = "dat.molloy2014", escalc = list(measure = "ZCOR", ri = "ri", ni = "ni"), mods = "~ meanage"),
  mcdaniel1994_zcor = list(data = "dat.mcdaniel1994", escalc = list(measure = "ZCOR", ri = "ri", ni = "ni"), mods = "~ type"),
  cohen1981_zcor  = list(data = "dat.cohen1981", escalc = list(measure = "ZCOR", ri = "ri", ni = "ni"), mods = NULL),
  pritz1997_pr    = list(data = "dat.pritz1997", escalc = list(measure = "PR", xi = "xi", ni = "ni"), mods = NULL),
  pritz1997_plo   = list(data = "dat.pritz1997", escalc = list(measure = "PLO", xi = "xi", ni = "ni"), mods = NULL),
  debruin2009_plo = list(data = "dat.debruin2009", escalc = list(measure = "PLO", xi = "xi", ni = "ni"), mods = "~ scq"),
  pagliaro1992_plo = list(data = "dat.pagliaro1992", escalc = list(measure = "PLO", xi = "xi", ni = "ni"), mods = "~ trt"),
  hart1999_irr    = list(data = "dat.hart1999", escalc = list(measure = "IRR", x1i = "x1i", t1i = "t1i", x2i = "x2i", t2i = "t2i"), mods = NULL),
  nielweise2007_or = list(data = "dat.nielweise2007", escalc = list(measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  li2007_or       = list(data = "dat.li2007", escalc = list(measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  lee2004_or      = list(data = "dat.lee2004", escalc = list(measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  egger2001_or    = list(data = "dat.egger2001", escalc = list(measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  graves2010_or   = list(data = "dat.graves2010", escalc = list(measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  hine1989_rd     = list(data = "dat.hine1989", escalc = list(measure = "RD", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  laopaiboon2015_rr = list(data = "dat.laopaiboon2015", escalc = list(measure = "RR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = NULL),
  linde2005_rr    = list(data = "dat.linde2005", escalc = list(measure = "RR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), mods = "~ dosage"),
  dorn2007_rr     = list(data = "dat.dorn2007", escalc = list(measure = "RR", ai = "x.a", n1i = "n.a", ci = "x.p", n2i = "n.p"), mods = NULL)
)

# Heterogeneity estimators covered by the rma.uni fixtures. GENQ/GENQM need
# user weights and get dedicated cases in phase 1.
methods <- c("EE", "DL", "HE", "HS", "HSk", "SJ", "ML", "REML", "EB", "PM", "PMM")
