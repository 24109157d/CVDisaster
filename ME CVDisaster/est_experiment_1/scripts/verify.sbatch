#!/bin/bash
#SBATCH --partition=h240
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=4:00:00
#SBATCH --job-name=est-1-verify
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/est-1-verify-%j.log

set -euo pipefail
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
DATA=/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN
REFERENCE="$REPO/../runs/est-4087311/Author CVDisaster/disaster_perception_mapping"
COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
export CVDISASTER_COMMIT
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${COMMIT}^{commit}")
RUN=/scratch/liuzeyu/heyinxi/runs/est-1-verify-${SLURM_JOB_ID}
mkdir "$RUN"
mkdir "$RUN/source"
git -C "$REPO" archive "$CVDISASTER_COMMIT" 'ME CVDisaster/est_experiment_1' | tar -x -C "$RUN/source"
cp -- "$0" "$RUN/submitted.sbatch"
export CUDA_VISIBLE_DEVICES=-1
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
"$ENV/bin/python" prepare.py --metadata "$DATA/02_Position/CVIAN_position.geojson" --est-train "$REFERENCE/est_train_60.csv" --est-test "$REFERENCE/est_test_60.csv" --est-config "$REFERENCE/est_config.json" --output "$RUN/manifest"
"$ENV/bin/python" validation_cases.py --manifest "$RUN/manifest" --data-root "$DATA" --weights /scratch/liuzeyu/heyinxi/pretrained/keras/models/convnext_tiny_notop.h5 --output "$RUN/cases.json"
if (( $# == 0 )); then
    set -- full
fi
for variant in "$@"; do
    "$ENV/bin/python" -u verify_model.py --cases "$RUN/cases.json" --variant "$variant" --output "$RUN/$variant" --gpus 0
done
