#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=cvdisaster-est
#SBATCH --mem=60000M
#SBATCH --time=1-00:00:00
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

# 在初始化 TensorFlow 前检查实际可见显卡，避免再次在被其他进程占用的卡上训练。
# 仅检查本作业可见的两张卡；打印 UUID 和 Slurm 映射，不能把逻辑编号当物理编号。
# 每卡至少保留 38000 MiB 空闲显存，并禁止已有计算进程；不修改或终止其他进程。
check_est_gpu_capacity() {
python - <<'PY'
import csv
import io
import os
import subprocess

def query(arguments):
    return list(csv.reader(io.StringIO(subprocess.check_output(
        ['nvidia-smi', *arguments, '--format=csv,noheader,nounits'], text=True))))

gpus = query(['--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free'])
assert len(gpus) == 2, f'应仅看到本作业的两张 GPU，实际信息：{gpus}'
visible_uuids = {row[1].strip() for row in gpus}
processes = query(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'])
occupied = [row for row in processes if row and row[0].strip() in visible_uuids]
print('EST_GPU_ALLOCATION', {key: os.environ.get(key) for key in (
    'SLURM_JOB_ID', 'CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS')}, flush=True)
print('EST_GPU_CAPACITY', gpus, 'COMPUTE_PROCESSES', occupied, flush=True)
assert not occupied, f'分配的 GPU 已有计算进程占用，停止启动：{occupied}'
assert all(int(row[5].strip()) >= 38000 for row in gpus), \
    f'分配的 GPU 空闲显存不足 38000 MiB，停止启动：{gpus}'
print('EST_GPU_CAPACITY_OK', flush=True)
PY
}
check_est_gpu_capacity

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
# 预检作者 GitHub 使用的 SGD API，不安装依赖或切换到论文中的 AdamW。
assert tf.keras.optimizers.SGD is not None
print('EST_TWO_GPU_SGD_API_OK', flush=True)

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

# 实测本轮恢复的 SGD 两卡更新及报告阶段的新批量预测入口。
# 测试 5 个样本的不完整 batch，覆盖 1649 个测试样本最后一批的 3/2 分卡情况。
# 使用独立微型模型，不加载数据、不改正式模型的初始化、优化器或随机种子。
with strategy.scope():
    probe = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    probe.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
probe_x = tf.ones((12, 2)).numpy()
probe_y = tf.zeros((12,), dtype=tf.int32).numpy()
probe.train_on_batch(probe_x, probe_y)
probe_predictions = probe.predict_on_batch(probe_x[:5])
assert probe_predictions.shape == (5, 3)
assert bool(tf.reduce_all(tf.math.is_finite(probe_predictions)).numpy())
print('EST_TWO_GPU_SGD_PREDICT_BATCH_OK', flush=True)
# 在正式训练前验证两卡 latest 的恢复、下一步 SGD 更新及 best 保留规则。
from check_est_checkpoints import verify_checkpoints
verify_checkpoints(strategy)
PY

# 作者 GitHub 基线：两卡 Tiny、仅 6:4、全局 batch 12、SGD、LR 0.01、无显式衰减。
# 全局 batch 不变，学习率和调度步数不因卡数变化而缩放；逐图展示仍使用主卡。
# 训练 200 轮、预热 10 个优化步；目录由 JobID 隔离，不覆盖任何旧结果。
# 正式分类报告基于完整测试集，estimation.txt 仅用于作者原有逐图展示。
# 前面的独立 TensorFlow 预检退出后再检查一次，避免预检期间出现的新占用被忽略。
check_est_gpu_capacity
# 默认从预训练权重开始；仅在明确指定旧输出目录时从 latest 续训。
# 旧结果复制到新 JobID 目录后继续保存，不覆盖或修改旧作业的 best/latest。
TRAIN_ARGS=(--tr-ratio 0.6 --backbone tiny --gpus 2)
if [[ -n "${CVDISASTER_EST_RESUME_FROM:-}" ]]; then
    test -f "$CVDISASTER_EST_RESUME_FROM/latest/current.json"
    TRAIN_ARGS+=(--resume-from "$CVDISASTER_EST_RESUME_FROM")
fi
python -u 2_cv.py "${TRAIN_ARGS[@]}"
test -s 0.6_bestCV.h5
python -u 3_estimation.py --tr-ratio 0.6 --backbone tiny
printf 'EST_FULL_TRAINING_AND_INFERENCE_COMPLETED\n'
