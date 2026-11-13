#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=04:00:00
#SBATCH --job-name=combined-1-evaluate
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/combined-1-evaluate-%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2
export TORCH_HOME=/scratch/liuzeyu/heyinxi/pretrained/torch
export HF_HUB_CACHE=/scratch/liuzeyu/heyinxi/pretrained/huggingface/hub
export HF_HUB_OFFLINE=1
export KERAS_HOME=/scratch/liuzeyu/heyinxi/pretrained/keras
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
export CVDISASTER_COMMIT
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${COMMIT}^{commit}")
SPLITS=${CVDISASTER_SPLITS:?必须提供共用固定划分目录}
GEO_RUN=${CVDISASTER_GEO_RUN:?必须提供本次Geo训练的output目录}
EST_RUN=${CVDISASTER_EST_RUN:?必须提供本次Est训练的output目录}
RUN=/scratch/liuzeyu/heyinxi/runs/combined-1-evaluate-${SLURM_JOB_ID}
GEO_ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc
EST_ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
BASE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
mkdir "$RUN"
mkdir "$RUN/source"
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/combined_experiment_1' 'Author CVDisaster/geolocalization' 'Author CVDisaster/disaster_perception_mapping' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
cd "$RUN/source/ME CVDisaster/combined_experiment_1"
nvidia-smi
export LD_LIBRARY_PATH="$GEO_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
"$GEO_ENV/bin/python" -u retrieve.py --splits "$SPLITS" --geo-run "$GEO_RUN" --output "$RUN/output"
export LD_LIBRARY_PATH="$EST_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
"$EST_ENV/bin/python" -u estimate.py --splits "$SPLITS" --est-run "$EST_RUN" --output "$RUN/output"
