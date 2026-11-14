#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=geo-partial-match
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/geo-partial-match-%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/scratch/liuzeyu/heyinxi/pretrained/huggingface/hub
export TORCH_HOME=/scratch/liuzeyu/heyinxi/pretrained/torch
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
GEO_ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc
COMMIT=${CVDISASTER_COMMIT:?必须指定已经同步的代码提交号}
COMMIT=$(git -C "$REPO" rev-parse --verify "${COMMIT}^{commit}")
export CVDISASTER_COMMIT="$COMMIT"
export PATH="$GEO_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$GEO_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MODE=${CVDISASTER_MODE:-train}
case "$MODE" in train|evaluate) ;; *) exit 2 ;; esac
RUN=/scratch/liuzeyu/heyinxi/runs/geo_partial_match_gate-${MODE}-${SLURM_JOB_ID}
mkdir "$RUN"
mkdir "$RUN/source"
git -C "$REPO" archive "$COMMIT" 'ME CVDisaster/geo_experiment_1' 'Author CVDisaster/geolocalization' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
cd "$RUN/source/ME CVDisaster/geo_experiment_1"
if [ "$MODE" = evaluate ]; then
    CHECKPOINT=${CVDISASTER_CHECKPOINT:?评价必须指定完整 checkpoint}
    python -u train_partial.py --mode evaluate --checkpoint "$CHECKPOINT" --output "$RUN/output"
else
    python -u train_partial.py --mode train --config configs/geo_partial_match_gate.json --output "$RUN/output"
fi
