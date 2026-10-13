"""The Python code blocks of the user manual (``docs/``), shared by the docs example test and ``bin/gallery.py``."""

import dataclasses
import pathlib
import re

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
