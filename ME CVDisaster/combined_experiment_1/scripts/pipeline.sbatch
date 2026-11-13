#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=combined_experiment_1
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/combined-experiment-1-%j.log

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
RUN=/scratch/liuzeyu/heyinxi/runs/combined-experiment-1-${SLURM_JOB_ID}
GEO_ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc
EST_ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
BASE_PATH=$PATH
BASE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
mkdir "$RUN"
mkdir "$RUN/source"
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/combined_experiment_1' 'Author CVDisaster/geolocalization' 'Author CVDisaster/disaster_perception_mapping' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
cd "$RUN/source/ME CVDisaster/combined_experiment_1"
printf 'COMMIT=%s\nRUN=%s\n' "$CVDISASTER_COMMIT" "$RUN"
nvidia-smi

export PATH="$GEO_ENV/bin:$BASE_PATH"
export LD_LIBRARY_PATH="$GEO_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
"$GEO_ENV/bin/python" -u check_environment.py --stage geo
export PATH="$EST_ENV/bin:$BASE_PATH"
export LD_LIBRARY_PATH="$EST_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
"$EST_ENV/bin/python" -u check_environment.py --stage est

export PATH="$GEO_ENV/bin:$BASE_PATH"
export LD_LIBRARY_PATH="$GEO_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
printf 'PIPELINE_STAGE=prepare_splits\n'
"$GEO_ENV/bin/python" -u prepare_splits.py --output "$RUN/splits"
printf 'PIPELINE_STAGE=train_geo\n'
"$GEO_ENV/bin/python" -u train_geo.py --splits "$RUN/splits" --output "$RUN/geo"

export PATH="$EST_ENV/bin:$BASE_PATH"
export LD_LIBRARY_PATH="$EST_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
printf 'PIPELINE_STAGE=train_est\n'
"$EST_ENV/bin/python" -u train_est.py --splits "$RUN/splits" --output "$RUN/est"

export PATH="$GEO_ENV/bin:$BASE_PATH"
export LD_LIBRARY_PATH="$GEO_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
printf 'PIPELINE_STAGE=retrieve\n'
"$GEO_ENV/bin/python" -u retrieve.py --splits "$RUN/splits" --geo-run "$RUN/geo" --output "$RUN/evaluation"

export PATH="$EST_ENV/bin:$BASE_PATH"
export LD_LIBRARY_PATH="$EST_ENV/lib${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"
printf 'PIPELINE_STAGE=estimate\n'
"$EST_ENV/bin/python" -u estimate.py --splits "$RUN/splits" --est-run "$RUN/est" --output "$RUN/evaluation"
test -s "$RUN/evaluation/result.json"
printf 'COMBINED_EXPERIMENT_1_PIPELINE_COMPLETED\n'
