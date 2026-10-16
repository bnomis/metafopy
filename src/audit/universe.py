"""The universe the audit enumerates: metafor's functions, S3 methods and datasets as dumped by ``r/universe.R``."""

import dataclasses
import json
import pathlib
import typing

import audit.config

Json = dict[str, typing.Any]


@dataclasses.dataclass
class Universe:
    doc: Json

    @classmethod
    def load(cls, path: pathlib.Path) -> Universe:
        doc: Json = json.loads(path.read_text())
        for entry in [*doc['functions'].values(), *doc['methods']]:
            if not isinstance(entry.get('body_choices'), dict):
                entry['body_choices'] = {}
            entry['chkclass'] = [c if isinstance(c, dict) else {} for c in entry.get('chkclass', [])]
        return cls(doc)

    @property
    def functions(self) -> dict[str, Json]:
        out: dict[str, Json] = self.doc['functions']
        return out

    @property
    def methods(self) -> list[Json]:
        out: list[Json] = self.doc['methods']
        return out

    @property
    def datasets(self) -> list[str]:
        out: list[str] = self.doc['datasets']
        return out

    @property
    def fixtures(self) -> dict[str, Json]:
        out: dict[str, Json] = self.doc['fixtures']
        return out

    def generics(self) -> list[str]:
        return sorted({m['generic'] for m in self.methods})

    def classes_of(self, fixture: str) -> list[str]:
        info = self.fixtures.get(fixture, {})
        return [str(c) for c in info.get('class', [])]

    def method_for(self, generic: str, classes: list[str]) -> Json | None:
        """The S3 method R dispatches to: the first class of the chain that has one."""
        by_class = {m['class']: m for m in self.methods if m['generic'] == generic}
        for cls in classes:
            if cls in by_class:
                return by_class[cls]
        return None

    def formals_of(self, name: str) -> Json | None:
        return self.functions.get(name)

    def draws(self, generic: str) -> bool:
        """True when the generic or any of its methods draws a figure."""
        if self.functions.get(generic, {}).get('draws'):
            return True
        return any(m.get('draws') for m in self.methods if m['generic'] == generic)

    def refuses(self, generic: str, classes: list[str]) -> bool:
        """True when R's own class checks (``.chkclass``) reject a fit of these classes for this generic."""
        method = self.method_for(generic, classes)
        if method is None:
            return True
        for check in method.get('chkclass', []):
            must = check.get('must', [])
            if must and not any(c in classes for c in must):
                return True
            if any(c in classes for c in check.get('notav', [])):
                return True
        return False


def snapshot_path() -> pathlib.Path:
    return audit.config.DATA_DIR / 'universe.json'
