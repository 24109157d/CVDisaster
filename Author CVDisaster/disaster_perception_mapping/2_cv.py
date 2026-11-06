import argparse
import csv
import gc
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
parser.add_argument('--gpus', type=int, choices=(2,), default=2,
                    help='本轮固定使用两张 GPU，全局 batch 12 由两卡共享')
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

# 单卡 batch 12 已发生显存不足，恢复两卡镜像训练；设备不足直接报错。
# batch 仍是两卡合计 12，每卡约 6 对，不按 GPU 数增加 batch 或学习率。
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) != 2:
    raise RuntimeError(f'Est 本轮要求恰好两张可见 GPU，实际为 {gpus}')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# GCViT resize_query=True 的双三次缩放在 TF 2.10 中需要 CPU 内核。
# 允许该算子按内核支持情况放置；模型变量仍在两张 GPU 上各保存一个副本。
tf.config.set_soft_device_placement(True)
strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0', '/GPU:1'])
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
# 按作者公开 GitHub 训练代码恢复基线：预热单位为优化步，不能乘每轮 batch 数。
# SGD 不显式增加 momentum、weight decay、梯度裁剪或参数衰减排除规则。
EPOCHS = 200
WARMUP_STEPS = 10
INIT_LR = 0.01
WAMRUP_LR = 0.001

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
    split_file = pathlib.Path(f'est_{split_name}_{round(TR_RATIO * 100)}.csv')
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
run_config = {
    'protocol': 'github-est-sgd', 'backbone': args.backbone,
    'train_ratio': TR_RATIO, 'split_seed': SEED, 'gpus': args.gpus,
    'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'optimizer': 'SGD',
    'base_lr': INIT_LR, 'warmup_lr': WAMRUP_LR, 'weight_decay': None,
    'momentum': 0.0, 'nesterov': False, 'steps_per_epoch': STEPS_PER_EPOCH,
    'warmup_steps': WARMUP_STEPS, 'total_steps': TOTAL_STEPS,
    'weight_decay_exclusions': [], 'splits': split_manifest,
}
pathlib.Path('est_config.json').write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
print('EST_CONFIG', json.dumps(run_config, ensure_ascii=False), flush=True)

# model
with strategy.scope():
    # 保留两卡所需的作用域适配；优化器调用与作者 GitHub 完全相同，使用默认 SGD 参数。
    lr_schedule = WarmUpCosine(
        learning_rate_base=INIT_LR,
        total_steps=TOTAL_STEPS,
        warmup_learning_rate=WAMRUP_LR,
        warmup_steps=WARMUP_STEPS,
    )
    optimizer = tf.keras.optimizers.SGD(lr_schedule)
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
    # 逐个检查可训练权重的两个镜像副本，确保双卡均承载模型变量。
    # CPU 专属缩放算子的放置不改变模型权重必须位于两张 GPU 的要求。
    for variable in model.trainable_variables:
        replicas = strategy.experimental_local_results(variable)
        if len(replicas) != 2 or any(
                f'GPU:{index}' not in replica.device
                for index, replica in enumerate(replicas)):
            raise RuntimeError(f'Est 权重未正确镜像到两张 GPU: {variable.name}')
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

# 训练结束后释放训练数据缓存；评估仍加载同一个最佳 checkpoint。
# 原循环反复 model.predict 会反复创建数据适配器；改用两卡策略支持的 predict_on_batch。
# 仅保留每批标签与三个类别的概率，评估样本、顺序、argmax 和汇总定义均不改变。
del trData
gc.collect()
test_labels = []
test_predictions = []
for (x0, x1), labels in vaData:
    probabilities = model.predict_on_batch((x0, x1))
    test_labels.append(labels.numpy())
    test_predictions.append(np.argmax(probabilities, axis=1))
yTrue = np.concatenate(test_labels, axis=0)
yPred = np.concatenate(test_predictions, axis=0)

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
