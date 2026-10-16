#!/usr/bin/env python3
"""Run the metafopy-versus-metafor audit: ``bin/metafor-audit.sh`` (after ``source bin/activate.sh``)."""

import os
import sys


def metafopy_source(argv: list[str]) -> str | None:
    """The ``--metafopy-src`` option, read before anything imports metafopy so that the audit can run against an older checkout."""
    for i, arg in enumerate(argv):
        if arg == '--metafopy-src' and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith('--metafopy-src='):
            return arg.split('=', 1)[1]
    return None


source = metafopy_source(sys.argv[1:])
if source:
    os.environ['METAFOPY_AUDIT_SRC'] = os.path.abspath(source)
    os.environ['PYTHONPATH'] = os.path.abspath(source) + os.pathsep + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, os.path.abspath(source))

import audit.cli  # noqa: E402

if __name__ == '__main__':
    sys.exit(audit.cli.main())
