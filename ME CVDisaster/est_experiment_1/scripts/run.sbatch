#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=5-00:00:00
#SBATCH --array=0-4%1
#SBATCH --job-name=est_experiment_1
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/est-experiment-1-%A_%a.log

set -euo pipefail
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
MANIFEST=${CVDISASTER_EST_MANIFEST:?必须提供已核验的固定清单目录}
export CVDISASTER_COMMIT
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${COMMIT}^{commit}")
RUN=/scratch/liuzeyu/heyinxi/runs/est-experiment-1-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir "$RUN"
mkdir "$RUN/source"
mkdir "$RUN/cache"
export TMPDIR="$RUN/cache"
mkdir -p "$RUN/cache/cuda/nvvm/libdevice"
test -s "$ENV/lib/libdevice.10.bc"
ln -s "$ENV/lib/libdevice.10.bc" "$RUN/cache/cuda/nvvm/libdevice/libdevice.10.bc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$RUN/cache/cuda"
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/est_experiment_1' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO/ME CVDisaster/est_experiment_1/.local-cache/dependencies${PYTHONPATH:+:$PYTHONPATH}"
cd "$RUN/source/ME CVDisaster/est_experiment_1"
nvidia-smi
"$ENV/bin/python" -m pip check
SEED=$("$ENV/bin/python" -c 'import os; from common import configuration; print(configuration()["seeds"][int(os.environ["SLURM_ARRAY_TASK_ID"])])')
"$ENV/bin/python" pipeline.py --manifest "$MANIFEST" --output "$RUN/output" --seeds "$SEED" --defer-statistics "$@"
