#!/bin/bash
#SBATCH --partition=hpxg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=verify-est-cache
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/verify-est-cache-%j.log

# 使用免费 CPU 分区和现有 Est 环境，仅验证缓存及预训练加载，不训练、不安装依赖。
set -euo pipefail
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
COMMIT=${CVDISASTER_COMMIT:?提交前必须 export 已同步的提交号}
RUN=/scratch/liuzeyu/heyinxi/est-install/cache-verify-${SLURM_JOB_ID}
mkdir "$RUN"
git -C "$REPO" archive "$COMMIT" scripts/prepare_est_cache.py scripts/verify_est_cache.py | tar -x -C "$RUN"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export KERAS_HOME=/scratch/liuzeyu/heyinxi/pretrained/keras
export MPLCONFIGDIR="$RUN/matplotlib"
"$ENV/bin/python" -B -u "$RUN/scripts/verify_est_cache.py" --output "$RUN/result.json"
