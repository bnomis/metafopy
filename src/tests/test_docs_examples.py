"""Run the Python examples in the user manual (``docs/``) so they cannot rot.

A page is run top to bottom in one namespace, like a notebook. A block whose first line is ``# docs: skip`` is not run
(use it for examples that take long to fit).
"""

import contextlib
import dataclasses
import io
import pathlib
import re

import matplotlib.pyplot
import pytest

import metafopy.core.rng

DOCS = pathlib.Path(__file__).resolve().parents[2] / 'docs'
FENCE = re.compile(r'^```python[ \t]*\n(.*?)^```[ \t]*$', re.DOTALL | re.MULTILINE)
SKIP = '# docs: skip'


@dataclasses.dataclass(frozen=True)
class Block:
    line: int
    code: str
    skip: bool


def python_blocks(text: str) -> list[Block]:
    return [Block(line=text.count('\n', 0, m.start()) + 2, code=m.group(1), skip=m.group(1).lstrip().startswith(SKIP)) for m in FENCE.finditer(text)]


PAGES = sorted(p for p in DOCS.rglob('*.md') if python_blocks(p.read_text()))


def test_pages_found() -> None:
    names = {p.name for p in PAGES}
    assert {'getting-started.md', 'models.md', 'analyses.md'} <= names


@pytest.mark.parametrize('page', PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_page_examples_run(page: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {'__name__': f'docs_{page.stem}'}
    rng = metafopy.core.rng.GLOBAL  # the examples call set_seed, which must not leak into other tests
    try:
        for block in python_blocks(page.read_text()):
            if block.skip:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    exec(compile(block.code, f'{page.relative_to(DOCS)}:{block.line}', 'exec'), namespace)
                except Exception as e:
                    raise AssertionError(f'{page.relative_to(DOCS)} line {block.line}: {type(e).__name__}: {e}') from e
    finally:
        matplotlib.pyplot.close('all')
        metafopy.core.rng.GLOBAL = rng
