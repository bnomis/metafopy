"""The ledger of known differences: findings that are accepted, each with its reason, matched mechanically.

``data/known.toml`` holds ``[[known]]`` entries: ``match`` (a glob on the case id, or a list of globs), ``kind``, ``path`` and
``detail`` (globs, default ``*``; ``path`` may be a list of globs, any of which matches), ``reason`` and, optionally, ``ref``.
A finding that matches an entry is known; any other is new. An entry that
matches no finding of a full run is stale and is reported, so the ledger cannot outlive the difference it records.
"""

import dataclasses
import fnmatch
import tomllib
import typing

import audit.config

Json = dict[str, typing.Any]


@dataclasses.dataclass
class Entry:
    match: str | list[str]
    kind: str
    path: str | list[str]
    detail: str
    reason: str
    ref: str
    used: int = 0

    def matches(self, case_id: str, finding: Json) -> bool:
        return (
            any(fnmatch.fnmatchcase(case_id, glob) for glob in ([self.match] if isinstance(self.match, str) else self.match))
            and fnmatch.fnmatchcase(finding['kind'], self.kind)
            and any(fnmatch.fnmatchcase(finding.get('path', ''), glob) for glob in ([self.path] if isinstance(self.path, str) else self.path))
            and fnmatch.fnmatchcase(finding.get('detail', ''), self.detail)
        )


@dataclasses.dataclass
class Ledger:
    entries: list[Entry]

    @classmethod
    def load(cls) -> Ledger:
        path = audit.config.DATA_DIR / 'known.toml'
        doc = tomllib.loads(path.read_text()) if path.exists() else {}
        entries = [Entry(e['match'], e.get('kind', '*'), e.get('path', '*'), e.get('detail', '*'), e['reason'], e.get('ref', '')) for e in doc.get('known', [])]
        return cls(entries)

    def classify(self, case_id: str, finding: Json) -> Entry | None:
        for entry in self.entries:
            if entry.matches(case_id, finding):
                entry.used += 1
                return entry
        return None

    def stale(self) -> list[Entry]:
        return [e for e in self.entries if e.used == 0]
