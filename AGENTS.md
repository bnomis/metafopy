# metafopy

Python port of the R package [metafor](https://www.metafor-project.org/) (reference version 5.2-1).

## Rules

Development follows the rules, commands and skills in <https://github.com/bnomis/agents> (not copied here). Read them before changing code.

## Environment

- `source bin/activate.sh`, then run `pytest`, `ruff` and `mypy` directly. No `uv run`.
- Python `>=3.14.7,<3.15`; dependencies and `uv.lock` are managed with `uv`. Do not edit `uv.lock` by hand or without instruction.
- Tooling: `ruff format`, `ruff check`, `mypy` (strict). Single quotes, line length 160.
- Tests live in `src/tests/`. Fixtures in `src/tests/reference/` are generated from R by `src/tests/r/*.R` (`LC_ALL=C.UTF-8 Rscript`); R metafor is the numeric oracle.

## Project conventions

- API mirrors metafor: `.` becomes `_`, same argument names, 0-based indices.
- Imports use full module paths (`import numpy`, not `import numpy as np`); no `from __future__`; no leading-underscore names. See `claude/rules/`.
- Domain errors live in `metafopy.errors`.
- Link the R source and papers in docstrings.

## Audit

`bin/metafor-audit.sh` runs the whole audit against R metafor in one command and writes `wrong-or-missing.md`; `bin/metafor-audit.sh --recall` checks that it
reproduces what earlier audits found. See `src/audit/README.md`. Fix a miss of the recall check in the harness (`src/audit/data/`), not in a report.
