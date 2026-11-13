import os
from pathlib import Path
import sys

from common import REPOSITORY, sha256

sys.path.insert(0, str(REPOSITORY / 'Author CVDisaster' / 'disaster_perception_mapping'))

import numpy as np
import tensorflow as tf
from gcvit import GCViTTiny
from est_checkpoints import EstCheckpoints


def check_pretrained_cache(params):
    source = Path(params['initial_weights']).resolve(strict=True)
    target = Path.home() / '.keras' / 'datasets' / 'gcvit_tiny_weights.h5'
    if (sha256(source) != '90bf9bfaafd086a831e92edbbfbba9450fd56786c2d39b3ac3f215d0cef54768'
            or target.resolve(strict=True) != source or not os.access(target.parent.parent, os.W_OK)):
        raise ValueError('GCViT预训练权重或Keras默认缓存映射不符合复现环境')


def build_model():
    subModelSVI = GCViTTiny(input_shape=(512, 1024, 3), pretrain=True, resize_query=True)
    subModelSVI.reset_classifier(num_classes=0, head_act=None)
    subModelSVI._name = 'svi'
    subModelSAT = GCViTTiny(input_shape=(512, 512, 3), pretrain=True, resize_query=True)
    subModelSAT.reset_classifier(num_classes=0, head_act=None)
    subModelSAT._name = 'sat'
    inpSVI = tf.keras.Input(shape=(512, 1024, 3))
    inpSAT = tf.keras.Input(shape=(512, 512, 3))
    featSVI = subModelSVI(inpSVI)
    featSAT = subModelSAT(inpSAT)
    concat = tf.keras.layers.Concatenate()([featSVI, featSAT])
    output = tf.keras.layers.Dense(units=64, activation=tf.keras.layers.LeakyReLU())(concat)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax)(output)
    return tf.keras.Model(inputs=[inpSVI, inpSAT], outputs=[output])


def configure_gpus(count):
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != count:
        raise RuntimeError(f'Est需要{count}张可见CUDA GPU，实际为{gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(True)


def load_image(path, width):
    return tf.image.resize(tf.io.decode_png(tf.io.read_file(path), channels=3), [512, width])


def make_dataset(samples, data_root, batch_size, satellites=None):
    root = Path(data_root)
    svi = [str(root / sample.raw_ground) for sample in samples]
    sat = [str(root / path) for path in satellites] if satellites is not None else [
        str(root / sample.satellite) for sample in samples]
    if len(svi) != len(sat):
        raise ValueError('街景与卫星输入数量不一致')
    labels = tf.constant([sample.label for sample in samples], dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((svi, sat, labels))
    dataset = dataset.map(lambda a, b, y: ((load_image(a, 1024), load_image(b, 512)), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_dataset(model, dataset):
    test_labels = []
    test_predictions = []
    for (x0, x1), labels in dataset:
        probabilities = model.predict_on_batch((x0, x1))
        if probabilities.shape != (len(labels), 3) or not np.isfinite(probabilities).all():
            raise ValueError('Est输出形状或数值不符合要求')
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        test_labels.append(labels.numpy())
        test_predictions.append(probabilities)
    return np.concatenate(test_labels, axis=0), np.concatenate(test_predictions, axis=0)


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
