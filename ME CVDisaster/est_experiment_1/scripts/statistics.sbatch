#!/bin/bash
#SBATCH --partition=h240
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=1:00:00
#SBATCH --job-name=est-1-statistics
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/est-1-statistics-%j.log

set -euo pipefail
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
ARRAY=${CVDISASTER_EST_ARRAY:?必须提供五个训练种子的数组作业号}
export CVDISASTER_COMMIT
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${COMMIT}^{commit}")
RUN=/scratch/liuzeyu/heyinxi/runs/est-1-statistics-${SLURM_JOB_ID}
mkdir "$RUN"
mkdir "$RUN/source" "$RUN/cache"
export TMPDIR="$RUN/cache"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/est_experiment_1' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
cd "$RUN/source/ME CVDisaster/est_experiment_1"
INDICES=()
for index in 0 1 2 3 4; do
    INDICES+=("/scratch/liuzeyu/heyinxi/runs/est-experiment-1-${ARRAY}_${index}/output/runs.json")
done
"$ENV/bin/python" analysis.py --run-index "${INDICES[@]}" --output "$RUN/statistics" --group sequence --repetitions 10000
