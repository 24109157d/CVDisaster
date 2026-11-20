#!/bin/bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est/bin/python
cd "$ROOT"
CACHE="$ROOT/.local-cache"
mkdir -p "$CACHE/wheels" "$CACHE/install-cache" "$CACHE/install-temporary"
export TMPDIR="$CACHE/install-temporary"
WEIGHTS=$("$PYTHON" -c 'from common import configuration; print(configuration()["initial_weights"])')
EXPECTED=$("$PYTHON" -c 'from common import configuration; print(configuration()["initial_sha256"])')
if [[ ! -f "$WEIGHTS" ]]; then
    mkdir -p "$(dirname -- "$WEIGHTS")"
    PARTIAL="$WEIGHTS.partial.$$"
    curl --fail --location --connect-timeout 15 --max-time 900 --output "$PARTIAL" https://storage.googleapis.com/tensorflow/keras-applications/convnext/convnext_tiny_notop.h5
    printf '%s  %s\n' "$EXPECTED" "$PARTIAL" | sha256sum --check
    mv -- "$PARTIAL" "$WEIGHTS"
fi
printf '%s  %s\n' "$EXPECTED" "$WEIGHTS" | sha256sum --check
"$PYTHON" -m pip download --index-url https://pypi.org/simple --timeout 30 --retries 0 --no-deps --only-binary=:all: --cache-dir "$CACHE/install-cache" --dest "$CACHE/wheels" pyproj==3.6.1
sha256sum "$CACHE/wheels/"*.whl > "$CACHE/wheels/SHA256SUMS"
