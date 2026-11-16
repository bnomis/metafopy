#!/usr/bin/env bash
# Run the group tests (or the platform probe) in a pinned Linux container, from a copy of the working tree.
set -euo pipefail

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$( cd -- "$THIS_DIR/.." &> /dev/null && pwd )

PYTHON_VERSION=3.14.8
ARCH=arm64
PROBE_DIR=
PYTEST_ARGS=()

usage() {
    cat <<'EOF'
Usage: bin/linux-tests.sh [--arch arm64|amd64] [--probe OUT_DIR] [-- PYTEST_ARGS...]

Build (or reuse) the image of bin/linux-tests.Dockerfile for linux/<arch>, stream the working tree into a container, run
`uv sync --locked --all-extras`, then pytest with PYTEST_ARGS, or with --probe bin/platform-probe.py writing
OUT_DIR/probe-linux-<arch>.json.

  --arch ARCH       arm64 (default, native on an Apple silicon Mac) or amd64 (emulated; CI's Linux is x86_64)
  --probe OUT_DIR   run the probe instead of pytest; OUT_DIR is created, and should be outside the repository
  -- PYTEST_ARGS    replaces the default test files (the failure groups' files)

Nothing in the checkout is written. uv's cache and the container's virtual environment live in the Docker volumes
metafopy-uv-cache and metafopy-venv-<arch>. uv.lock is never changed.
Exit status: pytest's or the probe's; 2 when Docker is not running, the image cannot be built or the arguments are wrong.
EOF
}

DEFAULT_TESTS=(
    src/tests/correctness/test_derived.py
    src/tests/correctness/test_properties.py
    src/tests/test_nchg.py
    src/tests/test_stepcon.py
    src/tests/test_ls_vif.py
    src/tests/test_audit4_numerics.py
    src/tests/test_rma_ls.py
    src/tests/test_selmodel.py
)

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --arch)
            [ $# -ge 2 ] || { echo "--arch needs arm64 or amd64" >&2; exit 2; }
            ARCH=$2; shift 2 ;;
        --probe)
            [ $# -ge 2 ] || { echo "--probe needs an output directory" >&2; exit 2; }
            PROBE_DIR=$2; shift 2 ;;
        --) shift; PYTEST_ARGS=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$ARCH" in
    arm64|amd64) ;;
    *) echo "--arch must be arm64 or amd64, not $ARCH" >&2; exit 2 ;;
esac

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running: start Docker Desktop" >&2
    exit 2
fi

IMAGE="metafopy-linux-tests:$ARCH"
if ! docker build --quiet --platform "linux/$ARCH" -f "$THIS_DIR/linux-tests.Dockerfile" -t "$IMAGE" "$THIS_DIR" >/dev/null; then
    echo "The image $IMAGE did not build" >&2
    exit 2
fi

MOUNT=()
if [ -n "$PROBE_DIR" ]; then
    mkdir -p "$PROBE_DIR"
    PROBE_DIR=$( cd -- "$PROBE_DIR" &> /dev/null && pwd )
    RUN=(uv run python bin/platform-probe.py --out "/probe/probe-linux-$ARCH.json")
    MOUNT=(-v "$PROBE_DIR:/probe")
else
    if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then
        PYTEST_ARGS=("${DEFAULT_TESTS[@]}")
    fi
    RUN=(uv run pytest --mpl "${PYTEST_ARGS[@]}")
fi

# The tree goes in by tar, so the host .venv, .audit and untracked data never enter the container. A deleted tracked file is skipped.
cd "$ROOT"
git ls-files -co --exclude-standard -z | while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done \
    | tar --null -T - -cf - \
    | docker run --rm -i --platform "linux/$ARCH" \
        -v metafopy-uv-cache:/uv-cache \
        -v "metafopy-venv-$ARCH:/work/.venv" \
        ${MOUNT[@]+"${MOUNT[@]}"} \
        -e UV_CACHE_DIR=/uv-cache \
        -e UV_LINK_MODE=copy \
        -e PYTHONDONTWRITEBYTECODE=1 \
        "$IMAGE" \
        bash -c 'set -e; bsdtar -xf -; uv sync --locked --all-extras --python "$0"; exec "${@:1}"' "$PYTHON_VERSION" "${RUN[@]}"
