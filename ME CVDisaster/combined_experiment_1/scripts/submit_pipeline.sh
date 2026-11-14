#!/bin/bash

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
unset CVDISASTER_REUSE_GEO_REFERENCE
if (( $# > 0 )); then
    [[ $# == 2 && "$1" == --reuse-geo-reference ]]
    [[ "$2" != /* && "$2" != *..* ]]
    test -s "$SCRIPT_DIR/../$2"
    export CVDISASTER_REUSE_GEO_REFERENCE=$2
fi
CVDISASTER_COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${CVDISASTER_COMMIT}^{commit}")
export CVDISASTER_COMMIT
mkdir -p "$REPO/.local-cache"
exec 9>"$REPO/.local-cache/combined-experiment-1-submit.lock"
flock -n 9
active_jobs=$(squeue --noheader --user="$(id -un)" --name=combined_experiment_1 --format=%A)
if [[ -n "$active_jobs" ]]; then
    printf '已有combined_experiment_1活动作业: %s\n' "$active_jobs" >&2
    exit 1
fi
unset SBATCH_GET_USER_ENV SLURM_GET_USER_ENV SBATCH_EXPORT SLURM_EXPORT_ENV
sbatch --parsable --export=ALL --comment="combined_experiment_1:$CVDISASTER_COMMIT" "$SCRIPT_DIR/pipeline.sbatch"
