#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
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
COMMIT=${CVDISASTER_COMMIT:?提交前必须 export CVDISASTER_COMMIT 并使用纯 --export=ALL}
RUN=/scratch/liuzeyu/heyinxi/runs/est-${SLURM_JOB_ID}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 权重保留在 scratch；Keras 2.10 get_file 的默认 home 缓存通过软链接映射到它。
test -s "$KERAS_HOME/datasets/gcvit_tiny_weights.h5"
test -d "$DATA/00_SVI"
test -d "$DATA/01_Satellite"
mkdir "$RUN"
git -C "$REPO" archive "$COMMIT" 'Author CVDisaster/disaster_perception_mapping' scripts/prepare_est_cache.py | tar -x -C "$RUN"
"$ENV/bin/python" -B "$RUN/scripts/prepare_est_cache.py" --check-only
cd "$RUN/Author CVDisaster/disaster_perception_mapping"
printf 'COMMIT=%s\nRUN=%s\nDATA=%s\n' "$COMMIT" "$RUN" "$DATA"
nvidia-smi

# 禁止静默回退至 CPU，要求 TensorFlow 完成一次实际 GPU 矩阵计算。
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) == 2, f'TensorFlow 应看到两张 GPU，实际为 {gpus}'
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
for index in range(2):
    with tf.device(f'/GPU:{index}'):
        x = tf.ones((16, 16))
        y = tf.matmul(x, x)
    assert float(tf.reduce_sum(y).numpy()) == 4096
    print('EST_GPU_OK', tf.__version__, y.device, flush=True)
# 预检现有 Keras 的 AdamW API，不安装依赖、不改用其他优化器。
assert tf.keras.optimizers.experimental.AdamW is not None
print('EST_TWO_GPU_ADAMW_API_OK', flush=True)

# 复现上次失败的 GPU 作用域内双三次缩放及反向传播，验证 CPU 专属算子可以执行。
# 上面的矩阵运算仍禁用软放置以严格检查 GPU；正式模型允许按算子内核分配设备。
tf.config.set_soft_device_placement(True)
with tf.device('/GPU:0'):
    resize_input = tf.Variable(tf.ones((1, 4, 4, 1)))
@tf.function
def resize_gradient():
    with tf.device('/GPU:0'):
        with tf.GradientTape() as tape:
            resized = tf.image.resize(resize_input, (8, 8), method='bicubic')
            loss = tf.reduce_sum(resized * resized)
        gradient = tape.gradient(loss, resize_input)
    return gradient
gradient = resize_gradient()
assert gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
print('EST_BICUBIC_FORWARD_BACKWARD_OK', flush=True)

# 验证两卡镜像策略及跨卡结果归并，提前暴露设备或通信配置错误。
strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0', '/GPU:1'])
@tf.function
def distributed_sum():
    values = strategy.run(lambda: tf.constant(1.0))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, values, axis=None)
assert strategy.num_replicas_in_sync == 2
assert float(distributed_sum().numpy()) == 2.0
print('EST_TWO_GPU_DISTRIBUTION_OK', flush=True)
PY

# 新版论文对比实验：两卡 Tiny、6:4、全局 batch 12、AdamW、LR 0.03、WD 0.05。
# 全局 batch 不变，学习率和调度步数不因卡数变化而缩放；逐图展示仍使用主卡。
# 训练 100 轮、预热 10 轮；目录由 JobID 隔离，不覆盖第一轮 SGD 结果。
# 正式分类报告基于完整测试集，estimation.txt 仅用于作者原有逐图展示。
python -u 2_cv.py --tr-ratio 0.6 --backbone tiny --gpus 2
test -s 0.6_bestCV.h5
python -u 3_estimation.py --tr-ratio 0.6 --backbone tiny
printf 'EST_FULL_TRAINING_AND_INFERENCE_COMPLETED\n'
