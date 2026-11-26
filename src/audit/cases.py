"""Cases: one call on one fixture, in a language-neutral form that both runners execute."""

import dataclasses
import json
import pathlib
import typing

import audit.config
import audit.harness_io

__all__ = [
    'Case',
    'load_fixtures',
    'shards',
    'write_cases',
]


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
    out: list[dict[str, typing.Any]] = json.loads(audit.harness_io.read_text(audit.config.DATA_DIR / 'fixtures.json', 'load the fixtures'))
    return out


def write_cases(path: pathlib.Path, cases: list[Case]) -> None:
    audit.harness_io.write_text(path, json.dumps([c.to_json() for c in cases]), 'write the cases')


def shards[T](items: list[T], n: int) -> list[list[T]]:
    return [items[i::n] for i in range(n)]
