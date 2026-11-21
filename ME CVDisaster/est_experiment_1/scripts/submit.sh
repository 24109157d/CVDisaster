#!/bin/bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
    printf '必须提供一个完整 Git 提交号\n' >&2
    exit 2
fi
COMMIT=$1
CURRENT=$(git -C "$ROOT" rev-parse HEAD)
if [[ "$CURRENT" != "$COMMIT" ]]; then
    printf '超算当前提交号与请求提交号不一致\n' >&2
    exit 2
fi
exec env -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SBATCH_EXPORT_FILE \
    -u SLURM_GET_USER_ENV -u SLURM_EXPORT_ENV \
    sbatch --parsable --export=NIL "$ROOT/scripts/run.sbatch" "$COMMIT"
