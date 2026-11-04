import argparse
import csv
import hashlib
import json
import numpy as np
import os
import pathlib
import tensorflow as tf

from gcvit import GCViTXXTiny, GCViTXTiny, GCViTTiny, GCViTSmall, GCViTBase, GCViTLarge
from sklearn.metrics import classification_report, precision_score, recall_score


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


# args
parser = argparse.ArgumentParser(description='cross view options')
parser.add_argument('--tr-ratio', type=float, required=True,
                    help='training ratio (0, 1)')
parser.add_argument('--backbone', type=str, default='Tiny',
                    help='sum the integers (default: find the max)')
parser.add_argument('--gpus', type=int, choices=(1,), default=1,
                    help='本轮论文配置固定使用一张 GPU')
args = parser.parse_args()
if args.backbone == 'XXTiny' or args.backbone == 'xxtiny':
    backbone = GCViTXXTiny
elif args.backbone == 'XTiny' or args.backbone == 'xtiny':
    backbone = GCViTXTiny
elif args.backbone == 'Tiny' or args.backbone == 'tiny':
    backbone = GCViTTiny
elif args.backbone == 'Small' or args.backbone == 'small':
    backbone = GCViTSmall
elif args.backbone == 'Base' or args.backbone == 'base':
    backbone = GCViTBase
elif args.backbone == 'Large' or args.backbone == 'large':
    backbone = GCViTLarge

# 本轮明确使用单卡；设备不符合要求直接报错，不自动改用 CPU 或 TPU。
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) != 1:
    raise RuntimeError(f'Est 本轮要求恰好一张可见 GPU，实际为 {gpus}')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_soft_device_placement(False)
strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
print("Number of Accelerators: ", strategy.num_replicas_in_sync)

# hyperparameters
TR_RATIO = args.tr_ratio
print('==================', TR_RATIO, '==================')
BATCH_SIZE = 12
SVI_IMG_H = 512
SVI_IMG_W = 1024
SVI_IMAGE_SIZE = [SVI_IMG_H, SVI_IMG_W]
SAT_IMG_H = 512
SAT_IMG_W = 512
SAT_IMAGE_SIZE = [SAT_IMG_H, SAT_IMG_W]
SEED = 626
# 新版论文明确规定 AdamW、100 轮、基础学习率 0.03、衰减 0.05 和 10 轮预热。
# 论文未规定预热起点，沿用作者 GitHub 的 0.001，不增加额外参数衰减排除规则。
EPOCHS = 100
WARMUP_EPOCHS = 10
INIT_LR = 0.03
WAMRUP_LR = 0.001
WEIGHT_DECAY = 0.05

# build dataset
dirSVI = pathlib.Path('/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN/00_SVI')
dirSAT = pathlib.Path('/scratch/liuzeyu/heyinxi/dataset/m1749324/CVIAN/01_Satellite')

imgCountSVI = len(list(dirSVI.glob('*/*.png')))
imgCountSAT = len(list(dirSAT.glob('*/*.png')))
assert(imgCountSVI == imgCountSAT)
imgCount = imgCountSVI

dirImgSVI = tf.data.Dataset.list_files(str(dirSVI/'*/*.png'), shuffle=False)
dirImgSVI = dirImgSVI.shuffle(imgCount, reshuffle_each_iteration=False, seed=SEED)
className = np.array(sorted([item.name for item in dirSVI.glob('*')]))

trSize = int(imgCount * TR_RATIO)
trData = dirImgSVI.take(trSize)
vaData = dirImgSVI.skip(trSize)

# 在读取图像前固化本次实际划分，使用相对路径以便与旧实验逐样本比较。
# 不改变作者的 shuffle 种子、输入排序或 take/skip 算法，也不额外设置全局随机种子。
split_manifest = {}
for split_name, split_data in [('train', trData), ('test', vaData)]:
    split_paths = [os.path.relpath(os.fsdecode(path), str(dirSVI))
                   for path in split_data.as_numpy_iterator()]
    split_file = pathlib.Path(f'est_{split_name}_50.csv')
    with split_file.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['svi_relative_path', 'label'])
        writer.writerows((path, pathlib.Path(path).parent.name) for path in split_paths)
    split_manifest[split_name] = {
        'count': len(split_paths), 'file': str(split_file),
        'sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
    }
    if split_name == 'test':
        test_paths = split_paths  # 与 vaData 顺序一致，用于最终逐样本预测的路径关联。

def GetLabel(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == className
    # Integer encode the label
    return tf.argmax(one_hot)

def LoadImg(img, h=SAT_IMG_H, w=SAT_IMG_W):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [h, w])

def ProcessPath(file_path):
    label = GetLabel(file_path)
    svi = LoadImg(tf.io.read_file(file_path), w=SVI_IMG_W)
    sat = LoadImg(tf.io.read_file(tf.strings.regex_replace(
        file_path,
        "00_SVI",
        "01_Satellite")))
    return (svi, sat), label

def ConfigureData(ds):
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
trData = trData.map(ProcessPath, num_parallel_calls=tf.data.AUTOTUNE)
vaData = vaData.map(ProcessPath, num_parallel_calls=tf.data.AUTOTUNE)
trData = ConfigureData(trData)
vaData = ConfigureData(vaData)
# batch() 保留最后一个不完整 batch，必须用实际 cardinality 换算轮数和优化步数。
STEPS_PER_EPOCH = int(trData.cardinality().numpy())
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
WARMUP_STEPS = STEPS_PER_EPOCH * WARMUP_EPOCHS
run_config = {
    'protocol': '2025-paper-est-adamw', 'backbone': args.backbone,
    'train_ratio': TR_RATIO, 'split_seed': SEED, 'gpus': args.gpus,
    'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'optimizer': 'AdamW',
    'base_lr': INIT_LR, 'warmup_lr': WAMRUP_LR, 'weight_decay': WEIGHT_DECAY,
    'warmup_epochs': WARMUP_EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH,
    'warmup_steps': WARMUP_STEPS, 'total_steps': TOTAL_STEPS,
    'weight_decay_exclusions': [], 'splits': split_manifest,
}
pathlib.Path('est_config.json').write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
print('EST_CONFIG', json.dumps(run_config, ensure_ascii=False), flush=True)

# model
with strategy.scope():
    # 单卡作用域内构建模型及优化器。使用现有 TensorFlow/Keras 2.10 自带 AdamW，
    # 其衰减更新为 lr * weight_decay * variable；关闭默认 XLA，沿用原训练执行方式。
    lr_schedule = WarmUpCosine(
        learning_rate_base=INIT_LR,
        total_steps=TOTAL_STEPS,
        warmup_learning_rate=WAMRUP_LR,
        warmup_steps=WARMUP_STEPS,
    )
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, jit_compile=False)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    subModelSVI = backbone(input_shape=(SVI_IMG_H, SVI_IMG_W, 3), pretrain=True, resize_query=True)
    subModelSVI.reset_classifier(num_classes=0, head_act=None)
    subModelSVI._name = 'svi'
    subModelSAT = backbone(input_shape=(SAT_IMG_H, SAT_IMG_W, 3), pretrain=True, resize_query=True)
    subModelSAT.reset_classifier(num_classes=0, head_act=None)
    subModelSAT._name = 'sat'
    # add new head
    inpSVI = tf.keras.Input(shape=(SVI_IMG_H, SVI_IMG_W, 3))
    inpSAT = tf.keras.Input(shape=(SAT_IMG_H, SAT_IMG_W, 3))
    # get feat
    featSVI = subModelSVI(inpSVI)
    featSAT = subModelSAT(inpSAT)
    # concat, mlp, and classification
    concat = tf.keras.layers.Concatenate()([featSVI, featSAT])
    output = tf.keras.layers.Dense(units=64, activation=tf.keras.layers.LeakyReLU())(concat)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax)(output)
    model = tf.keras.Model(inputs=[inpSVI, inpSAT], outputs=[output])

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
# model.summary()

ckpt_path = str(TR_RATIO) + '_bestCV.h5'
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

history = model.fit(trData,
    validation_data=vaData,
    epochs=EPOCHS,
    callbacks=[ckpt_callback])

model.load_weights(ckpt_path)

yTrue = np.concatenate([y for (_, _), y in vaData], axis=0)
yPred = np.argmax(np.concatenate([model.predict((x0, x1)) for (x0, x1), _ in vaData], axis=0), axis=1)

print(classification_report(yTrue, yPred, target_names=['minor', 'moderate', 'severe'], digits=5))
overallP = precision_score(yTrue, yPred, average='macro')
overallR = recall_score(yTrue, yPred, average='macro')
print('Overall Precision:', overallP, '\nOverall Recall:', overallR)

# 分类报告以完整测试集为准；保存逐样本预测及全部汇总，保留 macro/weighted F1 的区别。
# 独立展示脚本可能因原图尺寸跳过样本，不能用其 estimation.txt 替代这里的结果。
report = classification_report(
    yTrue, yPred, target_names=['minor', 'moderate', 'severe'], output_dict=True)
if len(test_paths) != len(yTrue) or len(yTrue) != len(yPred):
    raise ValueError('测试清单、真实标签及预测数量不一致，拒绝输出截断的逐图结果')
report['best_epoch'] = int(np.argmax(history.history['val_accuracy'])) + 1
report['checkpoint'] = ckpt_path
pathlib.Path('est_metrics.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
with pathlib.Path('est_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['svi_relative_path', 'true_label', 'predicted_label'])
    writer.writerows(zip(test_paths, yTrue.tolist(), yPred.tolist()))
