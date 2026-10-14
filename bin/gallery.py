"""Render the figures of the plot gallery (docs/guide/gallery.md) with metafopy.

Runs the Python blocks of the page in one namespace, like the docs example test, with the current directory set to
docs/images/gallery/py, so the ``savefig`` calls of the blocks write the images there. The R figures they are compared with
are rendered by src/tests/r/gallery.R.

Usage (after ``source bin/activate.sh``):  python bin/gallery.py
"""

import argparse
import contextlib
import io
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'tests'))

import docsblocks  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')

import matplotlib.pyplot  # noqa: E402

PAGE = docsblocks.DOCS / 'guide' / 'gallery.md'
OUT = docsblocks.DOCS / 'images' / 'gallery' / 'py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='gallery.py',
        description='Render the figures of docs/guide/gallery.md into docs/images/gallery/py.',
    )
    parser.add_argument(
        '--page',
        type=pathlib.Path,
        default=PAGE,
        help='Markdown page whose Python blocks are run',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)
    namespace: dict[str, object] = {'__name__': 'gallery'}
    for block in docsblocks.python_blocks(args.page.read_text()):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(block.code, f'{args.page.name}:{block.line}', 'exec'), namespace)
        matplotlib.pyplot.close('all')
    print(f'wrote {len(list(OUT.glob("*.png")))} figures to {OUT}')


if __name__ == '__main__':
    main()
