#!/bin/bash
#SBATCH --partition=h240
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=04:00:00
#SBATCH --job-name=est_satellite_noise_cpu
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/est-satellite-noise-cpu-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=1
export KERAS_HOME=/scratch/liuzeyu/heyinxi/pretrained/keras
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
COMMIT=${CVDISASTER_COMMIT:?必须指定已经推送到GitHub并由超算拉取的提交号}
export CVDISASTER_COMMIT
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse --verify "${COMMIT}^{commit}")
REFERENCE=${CVDISASTER_REFERENCE_RUN:-/scratch/liuzeyu/heyinxi/runs/combined-experiment-1-4308780}
DIAG=/scratch/liuzeyu/heyinxi/diagnostics/combined-4308780/noise-${SLURM_JOB_ID}
mkdir "$DIAG"
mkdir "$DIAG/source"
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/combined_experiment_1/diagnostics' | tar -x -C "$DIAG/source"
cp -- "$0" "$DIAG/submitted.sbatch"
"$ENV/bin/python" -u "$DIAG/source/ME CVDisaster/combined_experiment_1/diagnostics/noise_satellite.py" --run "$REFERENCE" --output "$DIAG/output" --seed 42
