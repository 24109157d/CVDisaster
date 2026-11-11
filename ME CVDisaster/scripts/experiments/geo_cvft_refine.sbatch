#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=cvdisaster-geo-cvft
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/geo-cvft-%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/scratch/liuzeyu/heyinxi/pretrained/huggingface/hub
export TORCH_HOME=/scratch/liuzeyu/heyinxi/pretrained/torch

# 每次只执行明确指定的操作，运行副本来自已同步的 Git 提交。
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
GEO_ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc
COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的 CVDISASTER_COMMIT}
MODE=${CVDISASTER_MODE:?必须指定 validate、train 或 evaluate}
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${COMMIT}^{commit}")
export CVDISASTER_COMMIT
COMMIT=$CVDISASTER_COMMIT
export PATH="$GEO_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$GEO_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "$MODE" in
    validate)
        RUN=/scratch/liuzeyu/heyinxi/runs/geo-cvft-validate-${SLURM_JOB_ID}
        ;;
    train)
        ARCHITECTURE=${CVDISASTER_ARCHITECTURE:?必须指定模型名称}
        SEED=${CVDISASTER_SEED:?必须指定训练 seed}
        case "$ARCHITECTURE" in geo_cvft_refine) ;; *) exit 2 ;; esac
        case "$SEED" in 42) ;; *) exit 2 ;; esac
        RUN=/scratch/liuzeyu/heyinxi/runs/${ARCHITECTURE}-seed${SEED}-${SLURM_JOB_ID}
        ;;
    evaluate)
        CHECKPOINT=${CVDISASTER_CHECKPOINT:?必须指定实验 checkpoint}
        test -s "$CHECKPOINT"
        RUN=/scratch/liuzeyu/heyinxi/runs/geo-cvft-evaluate-${SLURM_JOB_ID}
        ;;
    *)
        printf '未知操作: %s\n' "$MODE" >&2
        exit 2
        ;;
esac

mkdir "$RUN"
mkdir "$RUN/source"
git -C "$REPO" archive "$COMMIT" 'ME CVDisaster' 'Author CVDisaster/geolocalization' scripts/prepare_geoloc_data.py requirements.txt requirements-geoloc.lock.txt | tar -x -C "$RUN/source"
export CVDISASTER_SLURM_SCRIPT="$RUN/submitted.sbatch"
cp -- "$0" "$CVDISASTER_SLURM_SCRIPT"
cd "$RUN/source/ME CVDisaster"
printf 'COMMIT=%s\nMODE=%s\nRUN=%s\n' "$COMMIT" "$MODE" "$RUN"
nvidia-smi

case "$MODE" in
    validate)
        python -u validate.py --output "$RUN/output"
        ;;
    train)
        python -u train.py --architecture "$ARCHITECTURE" --seed "$SEED" --output "$RUN/output"
        ;;
    evaluate)
        python -u evaluate.py --checkpoint "$CHECKPOINT" --output "$RUN/output"
        ;;
esac
