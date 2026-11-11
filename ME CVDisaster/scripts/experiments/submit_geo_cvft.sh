#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
CVDISASTER_COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${CVDISASTER_COMMIT}^{commit}")
export CVDISASTER_COMMIT

# 训练参数通过当前提交进程的环境传递给 Slurm。
export CVDISASTER_MODE=train
export CVDISASTER_ARCHITECTURE=geo_cvft_refine
export CVDISASTER_SEED=42
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV SBATCH_EXPORT SLURM_EXPORT_ENV

active_jobs=$(squeue --noheader --user="$(id -un)" --name=geo-cvft-s42 --format=%A)
if [[ -n "$active_jobs" ]]; then
    printf '已有同名活动作业: %s\n' "$active_jobs" >&2
    exit 1
fi

exec sbatch --parsable --export=ALL --job-name=geo-cvft-s42 "$SCRIPT_DIR/geo_cvft_refine.sbatch"
