"""Every public callable has an inspectable signature (regression: rma_uni's ScaleSpec annotation)."""

import subprocess
import sys

SCRIPT = """
import inspect
import metafopy

bad = []
for name in metafopy.__all__:
    obj = getattr(metafopy, name)
    if callable(obj):
        try:
            inspect.signature(obj)
        except Exception as exc:
            bad.append(f'{name}: {exc!r}')
    if inspect.isclass(obj):
        for attr, member in vars(obj).items():
            if callable(member) and not attr.startswith('_'):
                try:
                    inspect.signature(member)
                except Exception as exc:
                    bad.append(f'{name}.{attr}: {exc!r}')
print('\\n'.join(bad))
"""


def test_public_signatures_inspectable_in_fresh_interpreter() -> None:
    # a fresh interpreter, so modules other tests import cannot mask a missing runtime import
    out = subprocess.run([sys.executable, '-c', SCRIPT], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ''
