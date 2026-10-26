#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --account=wanyi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=cvdisaster-geoloc
#SBATCH --mem=60000M
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/liuzeyu/heyinxi/logs/geoloc-%j.log

# 任何前置步骤失败都立即退出，避免缺少 GPU 或权重时转为其他训练方式。
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TORCH_HOME=/scratch/liuzeyu/heyinxi/pretrained/torch
export HF_HUB_CACHE=/scratch/liuzeyu/heyinxi/pretrained/huggingface/hub
export HF_HUB_OFFLINE=1

# 提交时必须传入已经同步的提交号；运行副本、日志和模型产物保存在仓库外。
REPO=/scratch/liuzeyu/heyinxi/CVDisaster
ENV=/scratch/liuzeyu/anaconda3/envs/cvdisaster_geoloc
DATA=/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN
WEIGHTS=/scratch/liuzeyu/heyinxi/pretrained/weights_e40_98.6830.pth
COMMIT=${CVDISASTER_COMMIT:?提交时必须通过 --export 传入 CVDISASTER_COMMIT}
RUN=/scratch/liuzeyu/heyinxi/runs/geoloc-${SLURM_JOB_ID}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -s "$WEIGHTS"
test -s "$HF_HUB_CACHE/models--timm--convnext_base.fb_in22k_ft_in1k_384/snapshots/2b2698d86523262ca8065d0957c7f80d963bb43f/model.safetensors"
test -s "$DATA/02_Position/CVIAN_position.geojson"
test -d "$DATA/00_SVI"
test -d "$DATA/01_Satellite"
mkdir "$RUN"
git -C "$REPO" archive "$COMMIT" 'Author CVDisaster/geolocalization' | tar -x -C "$RUN"
cd "$RUN/Author CVDisaster/geolocalization"
ln -s "$WEIGHTS" weights_e40_98.6830.pth
printf 'COMMIT=%s\nRUN=%s\nDATA=%s\n' "$COMMIT" "$RUN" "$DATA"
nvidia-smi

# 先验证分配的 GPU 和实际 CUDA 运算，失败时不启动训练。
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Slurm 作业内 PyTorch 无法使用 GPU'
assert torch.cuda.device_count() == 1, '本任务应只分配一张 GPU'
x = torch.ones((16, 16), device='cuda')
assert (x @ x).sum().item() == 4096
print('GEOLOC_GPU_OK', torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY

# 本次原始数据尚未预处理，先执行作者的图像对齐和划分生成脚本。
# 不覆盖已存在的划分，避免误用重复提交生成的新划分或覆盖其他实验输入。
test ! -e "$DATA/train_subarea5.csv"
test ! -e "$DATA/val_subarea5.csv"
python -u 0_prepare_splits_and_data.py
test -s "$DATA/train_subarea5.csv"
test -s "$DATA/val_subarea5.csv"

# 保留作者默认：ConvNeXt Base、subarea5、batch 128、10 轮，每轮评估。
python -u 1_train_cvdisaster.py
printf 'GEOLOC_FULL_TRAINING_COMPLETED\n'
