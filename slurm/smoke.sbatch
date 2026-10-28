#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cv-smoke
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/smoke-%x-%j.log

# 用户已批准单 batch 测试；只有手动 sbatch 才会启动，不自动提交任何后续作业。
set -euo pipefail
MODULE=${1:?必须指定 geoloc 或 est}
case "$MODULE" in geoloc|est) ;; *) exit 2 ;; esac
COMMIT=${CVDISASTER_COMMIT:?确认后提交时必须指定已同步的 Git 提交号}
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_${MODULE}
RUN=/scratch/liuzeyu/heyinxi/smoke-tests/${MODULE}-${SLURM_JOB_ID}

# 每个任务独立创建运行副本和缓存；不在源码仓库中写 pyc 或测试产物。
mkdir -p /scratch/liuzeyu/heyinxi/smoke-tests
mkdir "$RUN"
mkdir "$RUN/code" "$RUN/cache"
git -C "$REPO" archive "$COMMIT" scripts/smoke 'Author CVDisaster/geolocalization' 'Author CVDisaster/disaster_perception_mapping' | tar -x -C "$RUN/code"
export CVDISASTER_COMMIT="$COMMIT"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export HF_HUB_CACHE=/scratch/liuzeyu/heyinxi/pretrained/huggingface/hub
export HF_HUB_OFFLINE=1
export KERAS_HOME=/scratch/liuzeyu/heyinxi/pretrained/keras
export XDG_CACHE_HOME="$RUN/cache"
export MPLCONFIGDIR="$RUN/cache/matplotlib"
export TMPDIR="$RUN/cache"
printf 'MODULE=%s\nCOMMIT=%s\nRUN=%s\n' "$MODULE" "$COMMIT" "$RUN"
nvidia-smi
cd "$RUN/code"

# Python 最长运行 27 分钟，为归档和退出记录预留时间；Slurm 总时限 30 分钟。
# 超时、OOM 或检查失败均保留非零退出码，不降分辨率重试，也不回退 CPU。
set +e
timeout --signal=TERM --kill-after=30s 27m "$ENV/bin/python" -B -u "scripts/smoke/${MODULE}.py" \
    --data-root /scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN \
    --output "$RUN/output"
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$RUN/exit-code.txt"
exit "$STATUS"
