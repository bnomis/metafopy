#!/usr/bin/env bash
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# shellcheck disable=SC1091
source "$THIS_DIR/activate.sh"

"$THIS_DIR/metafor-audit.py" "$@"
