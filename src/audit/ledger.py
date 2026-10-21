"""The ledger of known differences: findings that are accepted, each with its reason, matched mechanically.

``data/known.toml`` holds ``[[known]]`` entries: ``match`` (a glob on the case id, or a list of globs), ``kind``, ``path`` and
``detail`` (globs, default ``*``; ``path`` may be a list of globs, any of which matches), ``reason`` and, optionally, ``ref``.
A finding that matches an entry is known; any other is new. An entry that
matches no finding of a full run is stale and is reported, so the ledger cannot outlive the difference it records.
A ``timeout`` entry for a case that is close to the time limit on its own can say ``load_dependent = true``: whether the case
times out depends on the machine and on the other shards, so a run in which it finished does not make the entry stale.

The globs are ``fnmatch`` globs, so ``[...]`` is a set of characters. A set that holds anything but letters, digits and ranges
(``R only ['FALSE']*``, ``*[class(dat)]*``) is a literal text that was not escaped: it matches one character, so the entry matches
almost nothing or almost everything. ``Ledger.loose_brackets`` lists those globs; write a literal ``[`` as ``[[]`` and ``]`` as ``[]]``.
"""

import dataclasses
import fnmatch
import re
import tomllib
import typing

import audit.config

Json = dict[str, typing.Any]

PLAIN_SET = re.compile(r'!?[A-Za-z0-9-]+|\[|\]')


def bracket_sets(glob: str) -> list[str]:
    """The ``[...]`` sets of an fnmatch glob, found the way ``fnmatch.translate`` finds them; a ``[`` without its ``]`` is literal."""
    sets = []
    i, n = 0, len(glob)
    while i < n:
        if glob[i] != '[':
            i += 1
            continue
        j = i + 1
        if j < n and glob[j] == '!':
            j += 1
        if j < n and glob[j] == ']':
            j += 1
        while j < n and glob[j] != ']':
            j += 1
        if j >= n:
            i += 1
            continue
        sets.append(glob[i + 1 : j])
        i = j + 1
    return sets


def globs_of(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


@dataclasses.dataclass
class Entry:
    match: str | list[str]
    kind: str
    path: str | list[str]
    detail: str
    reason: str
    ref: str
    load_dependent: bool = False
    used: int = 0

    def globs(self) -> list[str]:
        return [*globs_of(self.match), self.kind, *globs_of(self.path), self.detail]

    def label(self) -> str:
        return f'{self.match} {self.kind}' + ('' if self.detail == '*' else f' {self.detail!r}')

    def matches(self, case_id: str, finding: Json) -> bool:
        return (
            any(fnmatch.fnmatchcase(case_id, glob) for glob in globs_of(self.match))
            and fnmatch.fnmatchcase(finding['kind'], self.kind)
            and any(fnmatch.fnmatchcase(finding.get('path', ''), glob) for glob in globs_of(self.path))
            and fnmatch.fnmatchcase(finding.get('detail', ''), self.detail)
        )


@dataclasses.dataclass
class Ledger:
    entries: list[Entry]

    @classmethod
    def load(cls) -> Ledger:
        path = audit.config.DATA_DIR / 'known.toml'
        doc = tomllib.loads(path.read_text()) if path.exists() else {}
        entries = [
            Entry(e['match'], e.get('kind', '*'), e.get('path', '*'), e.get('detail', '*'), e['reason'], e.get('ref', ''), e.get('load_dependent', False))
            for e in doc.get('known', [])
        ]
        return cls(entries)

    def classify(self, case_id: str, finding: Json) -> Entry | None:
        for entry in self.entries:
            if entry.matches(case_id, finding):
                entry.used += 1
                return entry
        return None

    def stale(self) -> list[Entry]:
        return [e for e in self.entries if e.used == 0 and not e.load_dependent]

    def loose_brackets(self) -> list[tuple[Entry, str]]:
        """The globs with a ``[...]`` set that is a literal text left unescaped (see the module docstring)."""
        return [(e, glob) for e in self.entries for glob in e.globs() if any(not PLAIN_SET.fullmatch(s) for s in bracket_sets(glob))]
