# Spec Index

| Spec | Path | Description |
|------|------|-------------|
| [Interactive charts](interactive/spec.md) | `src/metafopy/interactive/` | Implemented: Vega-Lite forest and funnel charts built from a fit, behind the `interactive` extra, used by the showcase notebook |
| [Auditme clean](auditme-clean/spec.md) | repository-wide | Implemented: zero unaccepted auditme hits, then the commit, pre-push and CI guards |
| [Audit shard survival](audit-shard-survival/spec.md) | `src/audit/engine.py`, `src/audit/pyrun.py` | Implemented: Python audit shards survive a dead process, and a shard of either language past the memory limit is killed and its case reported |
| [Scripts shard survival](scripts-shard-survival/spec.md) | `src/audit/scripts.py`, `src/audit/r/runscripts.R` | Implemented: scripts-step shards of either language survive a dead or oversized process, restarting at the next script |
| [R oracle pin](r-oracle-pin/spec.md) | `src/audit/steps.py`, `src/audit/r/setup.R`, `src/audit/data/` | Implemented: the whole R library is pinned in `data/r-versions.json`, `check-r` refuses a different one, and recall items the oracle moved are counted apart |
