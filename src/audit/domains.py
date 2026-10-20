"""Domains and recipes: the hand-written part of case generation, read from ``data/domains.toml`` and ``data/recipes.toml``."""

import dataclasses
import json
import tomllib
import typing

import audit.config

Json = dict[str, typing.Any]


@dataclasses.dataclass
class Domains:
    args: dict[str, list[typing.Any]]
    skip: dict[str, str]
    per_function: dict[str, Json]
    args_alias: dict[str, str]
    graphics: list[str]
    options: dict[str, list[typing.Any]]
    options_skip: dict[str, str]
    companions: dict[str, Json]

    @classmethod
    def load(cls) -> Domains:
        doc = tomllib.loads((audit.config.DATA_DIR / 'domains.toml').read_text())
        return cls(
            args=doc.get('args', {}),
            skip=doc.get('skip', {}),
            per_function=doc.get('fns', {}),
            args_alias=doc.get('alias', {}),
            graphics=doc.get('graphics', {}).get('names', []),
            options=doc.get('options', {}),
            options_skip=doc.get('options_skip', {}),
            companions=doc.get('companions', {}),
        )

    def values(self, function: str, arg: str) -> list[typing.Any] | None:
        override = self.per_function.get(function, {}).get('args', {})
        if arg in override:
            return typing.cast(list[typing.Any], override[arg])
        return self.args.get(arg)

    def skipped(self, function: str, arg: str) -> str | None:
        local = self.per_function.get(function, {}).get('skip', {})
        return local.get(arg) or self.skip.get(arg)


@dataclasses.dataclass
class Recipes:
    doc: Json

    @classmethod
    def load(cls) -> Recipes:
        return cls(tomllib.loads((audit.config.DATA_DIR / 'recipes.toml').read_text()))

    @property
    def fit_functions(self) -> list[str]:
        out: list[str] = self.doc.get('fit_functions', [])
        return out

    def generic_calls(self, generic: str, fixture: str = '') -> list[Json]:
        """The argument sets of a generic's baseline cases on one fixture: the base alone, then the base with each extra."""
        entry = self.doc.get('generics', {}).get(generic, {})
        base = self.base_args(generic, fixture)
        return [base] + [{**base, **extra} for extra in entry.get('extra', [])]

    def pre(self, generic: str, fixture: str = '') -> list[Json]:
        """Calls that run first and leave something behind for the generic (a plot for addpoly to add to); a fit can have its own."""
        entry = self.doc.get('generics', {}).get(generic, {})
        out: list[Json] = entry.get('by_fit', {}).get(fixture, {}).get('pre', entry.get('pre', []))
        return out

    def base_args(self, generic: str, fixture: str = '') -> Json:
        entry = self.doc.get('generics', {}).get(generic, {})
        base: Json = {**entry.get('base', {}), **{k: v for k, v in entry.get('by_fit', {}).get(fixture, {}).items() if k != 'pre'}}
        return base

    def standalone(self) -> dict[str, Json]:
        out: dict[str, Json] = self.doc.get('standalone', {})
        return out

    def emmeans(self) -> dict[str, Json]:
        """Calls of the emmeans package on reference grids of metafor fits (``[emmeans.LABEL]``: ``call`` is a call spec, ``fn`` its function)."""
        out: dict[str, Json] = self.doc.get('emmeans', {})
        return out

    def patterns(self) -> dict[str, Json]:
        """Families of functions that share one recipe (``transf.*``): the arguments each function takes from the shared set are passed."""
        out: dict[str, Json] = self.doc.get('patterns', {})
        return out

    def unprobed(self) -> dict[str, str]:
        out: dict[str, str] = self.doc.get('unprobed', {})
        return out


def refs_in(value: typing.Any) -> list[str]:
    """The fixtures a JSON argument value refers to."""
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == '$ref':
                found.append(str(v))
            elif k == '$col':
                found.append(str(v[0]))
            else:
                found.extend(refs_in(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(refs_in(v))
    return found


def label(value: typing.Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).replace('/', '|')[:60]
