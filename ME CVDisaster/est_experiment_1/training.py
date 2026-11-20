import math
import os
from pathlib import Path
import uuid

import numpy as np
import tensorflow as tf

from common import read_json, sha256, write_json


class AccumulatingModel(tf.keras.Model):
    def __init__(self, functional, samples, global_batch=12):
        super().__init__(inputs=functional.input, outputs=functional.output, name=functional.name)
        self.samples = samples
        self.global_batch = global_batch

    def train_step(self, data):
        x, y = data
        gate = self.get_layer('uncertainty_gate')
        gate.progress.assign(tf.cast(self.optimizer.iterations, tf.float32) / gate.steps_per_epoch)
        replicas = self.distribute_strategy.num_replicas_in_sync
        local_count = tf.shape(y['final'])[0]
        context = tf.distribute.get_replica_context()
        global_count = context.all_reduce('sum', local_count)
        tf.debugging.assert_less_equal(global_count, self.global_batch)

        def body(index, *accumulated):
            current_x = {key: value[index:index + 1] for key, value in x.items()}
            current_y = {key: value[index:index + 1] for key, value in y.items()}
            with tf.GradientTape() as tape:
                prediction = self(current_x, training=True)
                loss = self.compiled_loss(current_y, prediction, regularization_losses=self.losses)
                loss *= tf.cast(replicas, loss.dtype) / tf.cast(global_count, loss.dtype)
                tf.debugging.check_numerics(loss, '训练损失出现非有限数值')
            gradients = tape.gradient(loss, self.trainable_variables)
            if any(value is None for value in gradients):
                missing = [v.name for v, g in zip(self.trainable_variables, gradients) if g is None]
                raise ValueError(f'可训练参数没有梯度: {missing}')
            self.compiled_metrics.update_state(current_y, prediction)
            return (index + 1, *[a + g for a, g in zip(accumulated, gradients)])

        initial = (tf.constant(0), *[tf.zeros_like(v) for v in self.trainable_variables])
        result = tf.while_loop(lambda index, *values: index < local_count, body, initial,
                               parallel_iterations=1)
        gradients = result[1:]
        for gradient in gradients:
            tf.debugging.check_numerics(gradient, '累计梯度出现非有限数值')
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {metric.name: metric.result() for metric in self.metrics}


class Checkpoints(tf.keras.callbacks.Callback):
    def __init__(self, model, directory, run_config, selection, resume=None):
        super().__init__()
        self.set_model(model)
        self.directory = Path(directory)
        self.run_config = run_config
        self.selection = selection
        self.best = -math.inf
        self.best_epoch = 0
        self.completed = 0
        self.best_file = self.directory / 'selected.h5'
        model.optimizer._create_all_weights(model.trainable_variables)
        weights = tf.train.Checkpoint(**{f'weight_{i}': v for i, v in enumerate(model.weights)})
        self.checkpoint = tf.train.Checkpoint(weights=weights, optimizer=model.optimizer)
        _ = self.checkpoint.save_counter
        self.signature = [[list(v.shape), v.dtype.name] for v in model.weights]
        self.latest = self.directory / 'latest'
        self.latest.mkdir()
        if resume is not None:
            self.restore(Path(resume))

    def restore(self, source):
        pointer = read_json(source / 'latest' / 'current.json')
        bundle = (source / 'latest' / pointer['directory']).resolve(strict=True)
        if bundle.parent != (source / 'latest').resolve():
            raise ValueError('续训快照超出指定目录')
        state = read_json(bundle / 'state.json')
        if (state['run_config'] != self.run_config or state['signature'] != self.signature
                or state['tensorflow'] != tf.__version__):
            raise ValueError('续训配置、环境、数据或权重结构不一致')
        self.checkpoint.restore(str(bundle / 'checkpoint')).assert_consumed()
        self.completed = state['completed_epochs']
        self.best = state['best_score'] if state['best_score'] is not None else -math.inf
        self.best_epoch = state['selected_epoch']
        expected_steps = self.completed * self.run_config['optimizer_steps_per_epoch']
        if int(self.model.optimizer.iterations.numpy()) != expected_steps:
            raise ValueError('续训 optimizer 步数与完整轮次不一致')
        if state['selected_sha256'] is not None:
            import shutil
            if sha256(bundle / 'selected.h5') != state['selected_sha256']:
                raise ValueError('续训 selected 权重指纹不一致')
            shutil.copyfile(bundle / 'selected.h5', self.best_file)

    def on_epoch_end(self, epoch, logs=None):
        metric_name = 'val_final_accuracy' if len(self.model.output) > 1 else 'val_accuracy'
        score = float(logs[metric_name]) if self.selection else 0.0
        if not np.isfinite(list(logs.values())).all():
            raise ValueError('训练或验证指标包含非有限数值')
        selected = score > self.best if self.selection else epoch + 1 == self.run_config['stop_epoch']
        if selected:
            temporary = self.directory / f'.selected-{uuid.uuid4().hex}.h5'
            self.model.save_weights(str(temporary))
            os.replace(temporary, self.best_file)
            self.best = score
            self.best_epoch = epoch + 1
        self.completed = epoch + 1
        bundle = self.latest / f'epoch-{self.completed:04d}-{uuid.uuid4().hex}'
        bundle.mkdir()
        self.checkpoint.write(str(bundle / 'checkpoint'))
        digest = None
        if self.best_file.exists():
            import shutil
            shutil.copyfile(self.best_file, bundle / 'selected.h5')
            digest = sha256(bundle / 'selected.h5')
        write_json(bundle / 'state.json', {
            'run_config': self.run_config, 'signature': self.signature,
            'tensorflow': tf.__version__, 'completed_epochs': self.completed,
            'optimizer_steps': int(self.model.optimizer.iterations.numpy()),
            'best_score': self.best if math.isfinite(self.best) else None,
            'selected_epoch': self.best_epoch, 'selected_sha256': digest})
        write_json(self.latest / 'current.json', {'directory': bundle.name}, replace=True)
        write_json(self.directory / 'progress.json', {'completed_epochs': self.completed,
                   'selected_epoch': self.best_epoch, 'logs': {key: float(value) for key, value in logs.items()}},
                   replace=True)
        # 每轮仅保留当前完整快照，清理范围限定为本作业 latest 的直接子目录。
        import shutil
        for previous in self.latest.glob('epoch-*'):
            if previous != bundle:
                if previous.resolve().parent != self.latest.resolve() or not previous.is_dir():
                    raise ValueError('快照清理路径无效')
                shutil.rmtree(previous)


def configure_devices(gpus):
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) != gpus:
        raise RuntimeError(f'要求 {gpus} 张 GPU，实际发现 {devices}')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.set_soft_device_placement(True)
    return tf.distribute.MirroredStrategy() if gpus else tf.distribute.get_strategy()
