"""Package metadata, and metafor's ``transf.*`` names as top-level functions.

``transf.ilogit`` is ``metafopy.transf.ilogit``; ``metafopy.transf_ilogit`` is the same function, so that metafor code
translates mechanically (``.`` becomes ``_``).
"""

import importlib.metadata

import metafopy.transf

__version__ = importlib.metadata.version('metafopy')

transf_abt = metafopy.transf.abt
transf_ahw = metafopy.transf.ahw
transf_arcsin = metafopy.transf.arcsin
transf_clestod = metafopy.transf.clestod
transf_dtobesd = metafopy.transf.dtobesd
transf_dtocles = metafopy.transf.dtocles
transf_dtocles_int = metafopy.transf.dtocles_int
transf_dtocliffd = metafopy.transf.dtocliffd
transf_dtolnor_logis = metafopy.transf.dtolnor_logis
transf_dtolnor_norm = metafopy.transf.dtolnor_norm
transf_dtomd = metafopy.transf.dtomd
transf_dtoovl = metafopy.transf.dtoovl
transf_dtorbis = metafopy.transf.dtorbis
transf_dtorpb = metafopy.transf.dtorpb
transf_dtou1 = metafopy.transf.dtou1
transf_dtou2 = metafopy.transf.dtou2
transf_dtou3 = metafopy.transf.dtou3
transf_exp_int = metafopy.transf.exp_int
transf_exp_mode = metafopy.transf.exp_mode
transf_iabt = metafopy.transf.iabt
transf_iabt_int = metafopy.transf.iabt_int
transf_iabt_mode = metafopy.transf.iabt_mode
transf_iahw = metafopy.transf.iahw
transf_iahw_int = metafopy.transf.iahw_int
transf_iahw_mode = metafopy.transf.iahw_mode
transf_iarcsin = metafopy.transf.iarcsin
transf_iarcsin_int = metafopy.transf.iarcsin_int
transf_iarcsin_mode = metafopy.transf.iarcsin_mode
transf_iirft = metafopy.transf.iirft
transf_ilogit = metafopy.transf.ilogit
transf_ilogit_int = metafopy.transf.ilogit_int
transf_ilogit_mode = metafopy.transf.ilogit_mode
transf_ipft = metafopy.transf.ipft
transf_ipft_hm = metafopy.transf.ipft_hm
transf_iprobit = metafopy.transf.iprobit
transf_iprobit_int = metafopy.transf.iprobit_int
transf_iprobit_mode = metafopy.transf.iprobit_mode
transf_irft = metafopy.transf.irft
transf_isqrt = metafopy.transf.isqrt
transf_lnortod_logis = metafopy.transf.lnortod_logis
transf_lnortod_norm = metafopy.transf.lnortod_norm
transf_lnortord = metafopy.transf.lnortord
transf_lnortorr = metafopy.transf.lnortorr
transf_lnortortet_digby = metafopy.transf.lnortortet_digby
transf_lnortortet_pearson = metafopy.transf.lnortortet_pearson
transf_logit = metafopy.transf.logit
transf_pft = metafopy.transf.pft
transf_probit = metafopy.transf.probit
transf_r2toz = metafopy.transf.r2toz
transf_rpbtod = metafopy.transf.rpbtod
transf_rpbtorbis = metafopy.transf.rpbtorbis
transf_rtod = metafopy.transf.rtod
transf_rtorpb = metafopy.transf.rtorpb
transf_rtoz = metafopy.transf.rtoz
transf_ztor = metafopy.transf.ztor
transf_ztor2 = metafopy.transf.ztor2
transf_ztor_int = metafopy.transf.ztor_int
transf_ztor_mode = metafopy.transf.ztor_mode
