#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=cvdisaster-est
#SBATCH --mem=60000M
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/est-%j.log

# 不加载案例中的 CUDA 12.2；使用已验证的 Est 环境内 CUDA 11.2/cuDNN 8.1。
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2
export KERAS_HOME=/scratch/liuzeyu/heyinxi/pretrained/keras

# 固定代码提交，将工作副本、权重与推理结果保存在独立的任务目录。
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_est
DATA=/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN
COMMIT=${CVDISASTER_COMMIT:?提交时必须通过 --export 传入 CVDISASTER_COMMIT}
RUN=/scratch/liuzeyu/heyinxi/runs/est-${SLURM_JOB_ID}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# GCViT 使用 Keras 默认 datasets 缓存目录；提前下载，计算节点不访问网络。
test -s "$KERAS_HOME/datasets/gcvit_tiny_weights.h5"
test -d "$DATA/00_SVI"
test -d "$DATA/01_Satellite"
mkdir "$RUN"
git -C "$REPO" archive "$COMMIT" 'Author CVDisaster/disaster_perception_mapping' | tar -x -C "$RUN"
cd "$RUN/Author CVDisaster/disaster_perception_mapping"
printf 'COMMIT=%s\nRUN=%s\nDATA=%s\n' "$COMMIT" "$RUN" "$DATA"
nvidia-smi

# 禁止静默回退至 CPU，要求 TensorFlow 完成一次实际 GPU 矩阵计算。
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) == 1, f'TensorFlow 应看到一张 GPU，实际为 {gpus}'
tf.config.set_soft_device_placement(False)
with tf.device('/GPU:0'):
    x = tf.ones((16, 16))
    y = tf.matmul(x, x)
assert float(tf.reduce_sum(y).numpy()) == 4096
print('EST_GPU_OK', tf.__version__, y.device, flush=True)
PY

# 保留作者完整双视角实验：Tiny、训练比例 0.5、batch 12、200 轮。
# 训练脚本末尾自动加载最佳权重并输出验证集分类指标，随后生成逐图推理结果。
python -u 2_cv.py --tr-ratio 0.5 --backbone tiny
test -s 0.5_bestCV.h5
python -u 3_estimation.py --backbone tiny
printf 'EST_FULL_TRAINING_AND_INFERENCE_COMPLETED\n'
