"""Griffe extension for the function reference: show the type behind each ``type X = ...`` alias of metafopy.

The API documentation then reads ``numpy.typing.ArrayLike | str | None`` where the source says ``Column | None``.
Griffe: https://mkdocstrings.github.io/griffe/guide/users/extending/
"""

import dataclasses

import griffe

MAX_DEPTH = 4


class ExpandAliases(griffe.Extension):
    """Replace metafopy type aliases in parameter and return annotations by their definition."""

    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: object) -> None:
        self.collection = loader.modules_collection
        for obj in self.walk(pkg):
            if isinstance(obj, griffe.Function):
                for param in obj.parameters:
                    param.annotation = self.expand(param.annotation, 0)
                obj.returns = self.expand(obj.returns, 0)

    def walk(self, obj: griffe.Object | griffe.Alias) -> list[griffe.Object]:
        found: list[griffe.Object] = []
        if obj.is_alias:
            return found
        found.append(obj)  # type: ignore[arg-type]
        for member in list(obj.members.values()):
            if not member.is_alias:
                found.extend(self.walk(member))
        return found

    def alias_value(self, expr: griffe.Expr) -> griffe.Expr | str | None:
        if not isinstance(expr, griffe.ExprName | griffe.ExprAttribute):
            return None
        path = expr.canonical_path if isinstance(expr, griffe.ExprName) else expr.last.canonical_path
        if not path.startswith('metafopy.'):
            return None
        try:
            target = self.collection[path]
        except KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError:
            return None
        if target.is_alias:
            try:
                target = target.final_target
            except griffe.AliasResolutionError, griffe.CyclicAliasError:
                return None
        return target.value if target.kind == griffe.Kind.TYPE_ALIAS else None

    def expand(self, expr: griffe.Expr | str | None, depth: int) -> griffe.Expr | str | None:
        if expr is None or isinstance(expr, str) or depth > MAX_DEPTH:
            return expr
        value = self.alias_value(expr)
        if value is not None:
            return self.expand(value, depth + 1)
        if isinstance(expr, griffe.ExprName | griffe.ExprAttribute):
            return expr
        changes = {}
        for field in dataclasses.fields(expr):
            current = getattr(expr, field.name)
            if isinstance(current, griffe.Expr):
                changes[field.name] = self.expand(current, depth)
            elif isinstance(current, list):
                changes[field.name] = [self.expand(item, depth) if isinstance(item, griffe.Expr) else item for item in current]
        return dataclasses.replace(expr, **changes) if changes else expr
