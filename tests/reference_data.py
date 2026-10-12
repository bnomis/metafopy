"""Access to the JSON fixtures generated from R metafor by tests/r/generate_fixtures.R."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

REFERENCE_DIR = Path(__file__).parent / "reference"


def case_names() -> list[str]:
    return sorted(p.stem for p in REFERENCE_DIR.glob("*.json"))


@cache
def load_case(name: str) -> dict[str, Any]:
    with (REFERENCE_DIR / f"{name}.json").open() as fh:
        return json.load(fh)
