"""Cases: one call on one fixture, in a language-neutral form that both runners execute."""

import dataclasses
import json
import pathlib
import typing

import audit.config


@dataclasses.dataclass
class Case:
    id: str
    axis: str
    call: dict[str, typing.Any]
    needs: list[str] = dataclasses.field(default_factory=list)
    print: bool = False
    meta: dict[str, str] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


def load_fixtures() -> list[dict[str, typing.Any]]:
    out: list[dict[str, typing.Any]] = json.loads((audit.config.DATA_DIR / 'fixtures.json').read_text())
    return out


def write_cases(path: pathlib.Path, cases: list[Case]) -> None:
    path.write_text(json.dumps([c.to_json() for c in cases]))


def shards[T](items: list[T], n: int) -> list[list[T]]:
    return [items[i::n] for i in range(n)]
