# Source this file: `source bin/activate.sh`
# Activates .venv and puts src/ on PYTHONPATH so metafopy is importable without installation.
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if [ -f "$_root/.venv/bin/activate" ]; then
    . "$_root/.venv/bin/activate"
fi
export PYTHONPATH="$_root/src${PYTHONPATH:+:$PYTHONPATH}"
unset _root
