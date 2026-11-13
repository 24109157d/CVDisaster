#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=combined-1-train
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/combined-1-train-%j.log

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
STAGE=${CVDISASTER_STAGE:?必须指定geo或est}
SPLITS=${CVDISASTER_SPLITS:?必须提供共用固定划分目录}
case "$STAGE" in
    geo) ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc ;;
    est) ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est ;;
    *) exit 2 ;;
esac
TRAIN_ARGS=()
if (( $# > 0 )); then
    [[ "$STAGE" == est && $# == 2 && "$1" == --resume-from ]]
    test -f "$2/latest/current.json"
    TRAIN_ARGS+=(--resume-from "$2")
fi
RUN=/scratch/liuzeyu/heyinxi/runs/combined-1-${STAGE}-${SLURM_JOB_ID}
mkdir "$RUN"
mkdir "$RUN/source"
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/combined_experiment_1' 'Author CVDisaster/geolocalization' 'Author CVDisaster/disaster_perception_mapping' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
cd "$RUN/source/ME CVDisaster/combined_experiment_1"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi
"$ENV/bin/python" -u "train_${STAGE}.py" --splits "$SPLITS" --output "$RUN/output" "${TRAIN_ARGS[@]}"
