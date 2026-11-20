#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60000M
#SBATCH --time=5-00:00:00
#SBATCH --job-name=est_experiment_1
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/est-experiment-1-%j.log

set -euo pipefail
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
COMMIT=${CVDISASTER_COMMIT:?必须提供已同步的代码提交号}
DATA=/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN
REFERENCE="$REPO/../runs/est-4087311/Author CVDisaster/disaster_perception_mapping"
export CVDISASTER_COMMIT
CVDISASTER_COMMIT=$(git -C "$REPO" rev-parse "${COMMIT}^{commit}")
RUN=/scratch/liuzeyu/heyinxi/runs/est-experiment-1-${SLURM_JOB_ID}
MANIFEST="$RUN/manifest"
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
export PYTHONPATH="$RUN/cache/dependencies${PYTHONPATH:+:$PYTHONPATH}"
cd "$RUN/source/ME CVDisaster/est_experiment_1"
nvidia-smi
ASSETS="$REPO/ME CVDisaster/est_experiment_1/.local-cache"
sha256sum --check "$ASSETS/wheels/SHA256SUMS"
"$ENV/bin/python" -m pip install --no-index --no-deps --target "$RUN/cache/dependencies" "$ASSETS/wheels/"pyproj-3.6.1-*.whl
"$ENV/bin/python" -c "import pyproj; assert pyproj.__version__ == '3.6.1', pyproj.__version__"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" prepare.py --metadata "$DATA/02_Position/CVIAN_position.geojson" --est-train "$REFERENCE/est_train_60.csv" --est-test "$REFERENCE/est_test_60.csv" --est-config "$REFERENCE/est_config.json" --output "$MANIFEST"
"$ENV/bin/python" validation_cases.py --manifest "$MANIFEST" --data-root "$DATA" --weights /scratch/liuzeyu/heyinxi/pretrained/keras/models/convnext_tiny_notop.h5 --output "$RUN/cases.json"
"$ENV/bin/python" verify_model.py --cases "$RUN/cases.json" --variant full --output "$RUN/verification" --gpus 2
"$ENV/bin/python" pipeline.py --manifest "$MANIFEST" --output "$RUN/output" --gpus 2
