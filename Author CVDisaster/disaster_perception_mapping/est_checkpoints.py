"""Est 的 best 测试权重及按完整轮次恢复的 latest 训练状态。"""

import json
import math
import os
from pathlib import Path
import shutil
import uuid

import tensorflow as tf


def _atomic_json(path, value):
    """同目录写临时文件后替换，使读取者只能看到完整的 JSON。"""
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class EstCheckpoints(tf.keras.callbacks.Callback):
    """best 用于最终测试；latest 同时保存权重、优化器及轮次状态。

    每轮发布一个完整目录，再原子更新 current.json；发布成功后删除旧目录。
    best 使用硬链接放入该目录，即使下一轮改写 best，旧 latest 仍能自洽恢复。
    只恢复完整轮次，未完成轮次重新训练；不承诺 Dropout 随机流逐位复现。
    """

    def __init__(self, model, config, best_path, resume_from=None):
        super().__init__()
        self.set_model(model)
        self.config = config
        self.best_path = Path(best_path).resolve()
        self.latest_dir = self.best_path.parent / 'latest'
        self.latest_dir.mkdir(exist_ok=True)
        self.current_file = self.latest_dir / 'current.json'
        if self.current_file.exists() or self.best_path.exists():
            raise FileExistsError('新作业输出目录必须为空，禁止覆盖已有 best/latest')

        # 固定 Keras 2.10 的 SGD 变量在恢复前创建，避免优化器状态延迟恢复后遗漏。
        # 仅创建变量，不执行梯度更新，因此不消耗正式训练步数或改变模型权重。
        with model.distribute_strategy.scope():
            model.optimizer._create_all_weights(model.trainable_variables)
            self.completed_epochs = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.best_score = tf.Variable(-math.inf, dtype=tf.float64, trainable=False)
            self.best_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
            # 显式记录全部模型权重，避免把 Keras 临时指标、数据迭代器和缓存纳入快照。
            weights = tf.train.Checkpoint(**{
                f'weight_{index}': variable for index, variable in enumerate(model.weights)
            })
            self.checkpoint = tf.train.Checkpoint(
                weights=weights, optimizer=model.optimizer,
                completed_epochs=self.completed_epochs,
                best_score=self.best_score, best_epoch=self.best_epoch)
            _ = self.checkpoint.save_counter

        self.signature = [[list(v.shape), v.dtype.name] for v in model.weights]
        if resume_from is not None:
            self._restore(Path(resume_from).resolve())

    @staticmethod
    def _published_bundle(latest_dir):
        """只读取显式发布的完整状态，不猜测临时文件或回退到 best。"""
        pointer = json.loads((latest_dir / 'current.json').read_text(encoding='utf-8'))
        bundle = (latest_dir / pointer['directory']).resolve()
        if bundle.parent != latest_dir.resolve() or not bundle.name.startswith('epoch-'):
            raise ValueError('latest 指针必须指向其目录内的 epoch 快照')
        return bundle

    def _restore(self, source_run):
        """从旧作业复制可测试的 best，并严格恢复同一训练配置下的 latest。"""
        bundle = self._published_bundle(source_run / 'latest')
        metadata = json.loads((bundle / 'state.json').read_text(encoding='utf-8'))
        if metadata['format_version'] != 1 or metadata['config'] != self.config:
            raise ValueError('续训配置或数据划分与 latest 不一致，禁止混用实验')
        if metadata['weights_signature'] != self.signature:
            raise ValueError('模型权重形状或 dtype 与 latest 不一致')
        if metadata['tensorflow_version'] != tf.__version__:
            raise ValueError('续训必须沿用保存时的 TensorFlow 环境')
        prefix = bundle / metadata['checkpoint']
        if prefix.parent != bundle or not prefix.name.startswith('ckpt-'):
            raise ValueError('检查点路径必须位于已发布的 latest 目录内')
        self.checkpoint.restore(str(prefix)).assert_consumed()
        if (int(self.completed_epochs.numpy()) != metadata['completed_epochs']
                or int(self.model.optimizer.iterations.numpy()) != metadata['optimizer_steps']
                or float(self.best_score.numpy()) != metadata['best_score']
                or int(self.best_epoch.numpy()) != metadata['best_epoch']):
            raise ValueError('latest 二进制状态和元数据不一致')
        if int(self.model.optimizer.iterations.numpy()) != (
                int(self.completed_epochs.numpy()) * self.config['steps_per_epoch']):
            raise ValueError('latest 不在完整轮次边界，禁止误用优化步数')
        if not 0 < int(self.best_epoch.numpy()) <= int(self.completed_epochs.numpy()) <= self.config['epochs']:
            raise ValueError('latest 保存的轮次范围无效')
        shutil.copyfile(bundle / 'best.h5', self.best_path)
        # 立即在新作业中发布恢复点；刚恢复就中断时也能从新目录继续，旧目录不被改写。
        self._save_latest()
        print('EST_RESUMED', source_run, 'completed_epochs=', self.initial_epoch,
              'optimizer_steps=', int(self.model.optimizer.iterations.numpy()),
              'best_epoch=', int(self.best_epoch.numpy()), flush=True)

    @property
    def initial_epoch(self):
        return int(self.completed_epochs.numpy())

    def on_epoch_end(self, epoch, logs=None):
        """测试指标仅决定 best；每个完整轮次均更新 latest，且不修改训练参数。"""
        score = float(logs['val_accuracy'])
        if not math.isfinite(score):
            raise ValueError('val_accuracy 非有限值，停止保存并直接暴露训练异常')
        if score > float(self.best_score.numpy()):
            temporary = self.best_path.with_name(f'.best-{uuid.uuid4().hex}.h5')
            self.model.save_weights(str(temporary))
            os.replace(temporary, self.best_path)
            self.best_score.assign(score)
            self.best_epoch.assign(epoch + 1)
            print('EST_BEST_SAVED', self.best_path, 'epoch=', epoch + 1,
                  'val_accuracy=', score, flush=True)
        self.completed_epochs.assign(epoch + 1)
        self._save_latest()

    def _save_latest(self):
        """先写完整快照，后切换指针；清理范围严格限定为本作业的旧快照。"""
        previous = self._published_bundle(self.latest_dir) if self.current_file.exists() else None
        token = uuid.uuid4().hex
        temporary = self.latest_dir / f'.pending-{token}'
        temporary.mkdir()
        prefix = self.checkpoint.save(str(temporary / 'ckpt'))
        # 同一运行文件系统上的硬链接不重复存储大权重，原子替换 best 不会改写旧链接。
        os.link(self.best_path, temporary / 'best.h5')
        metadata = {
            'format_version': 1, 'config': self.config,
            'tensorflow_version': tf.__version__, 'weights_signature': self.signature,
            'checkpoint': Path(prefix).name,
            'completed_epochs': self.initial_epoch,
            'optimizer_steps': int(self.model.optimizer.iterations.numpy()),
            'best_score': float(self.best_score.numpy()),
            'best_epoch': int(self.best_epoch.numpy()),
        }
        _atomic_json(temporary / 'state.json', metadata)
        published = self.latest_dir / f'epoch-{self.initial_epoch:04d}-{token}'
        temporary.rename(published)
        _atomic_json(self.current_file, {'directory': published.name})
        if previous is not None:
            if previous.parent != self.latest_dir.resolve() or not previous.name.startswith('epoch-'):
                raise ValueError('拒绝清理 latest 目录之外的路径')
            shutil.rmtree(previous)
        print('EST_LATEST_SAVED', published, 'epoch=', self.initial_epoch,
              'optimizer_steps=', metadata['optimizer_steps'], flush=True)
