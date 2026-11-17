"""Strict mkdocs builds abort on an unresolvable docstring link.

Each Markdown link in a docstring must resolve, from ``docs/reference/``, to a file under ``docs/`` and an existing heading.
"""

import ast
import pathlib
import re
import urllib.parse

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
PAGE_DIR = DOCS / 'reference'
LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)\)')
SCHEME = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*:')
HEADING = re.compile(r'^#{1,6}[ \t]+(.*?)[ \t#]*$', re.MULTILINE)


def slug(heading: str) -> str:
    return re.sub(r' ', '-', re.sub(r'[^\w\- ]', '', heading).strip().lower())


def anchors(path: pathlib.Path) -> set[str]:
    return {slug(m.group(1)) for m in HEADING.finditer(path.read_text(encoding='utf-8'))}


def docstring_links() -> list[tuple[str, str]]:
    found = []
    for source in sorted((ROOT / 'src' / 'metafopy').rglob('*.py')):
        tree = ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
                continue
            first = node.body[0]
            if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
                continue
            for m in LINK.finditer(first.value.value):
                target = m.group(1)
                if SCHEME.match(target) or target.startswith('#'):
                    continue
                line = first.lineno + first.value.value.count('\n', 0, m.start())
                found.append((f'{source.relative_to(ROOT)}:{line}', target))
    return found


def problem(target: str) -> str | None:
    parts = urllib.parse.urlsplit(target)
    resolved = (PAGE_DIR / urllib.parse.unquote(parts.path)).resolve()
    if not resolved.is_relative_to(DOCS) or not resolved.is_file():
        return f'{parts.path} does not resolve from docs/reference/ to a file under docs/'
    if parts.fragment and resolved.suffix == '.md' and parts.fragment not in anchors(resolved):
        return f'#{parts.fragment} is not a heading of {resolved.relative_to(ROOT)}'
    return None


@pytest.mark.parametrize(('where', 'target'), docstring_links())
def test_docstring_link_resolves(where: str, target: str) -> None:
    assert problem(target) is None, f'{where}: {target}: {problem(target)}'
