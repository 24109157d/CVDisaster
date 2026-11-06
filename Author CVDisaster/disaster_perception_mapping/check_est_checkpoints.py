"""在计算节点验证真实 TensorFlow 两副本的保存、恢复及下一步更新。"""

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

from est_checkpoints import EstCheckpoints


def verify_checkpoints(strategy):
    """使用微型模型验证状态闭环，不读取正式数据或改变正式训练进程。"""
    source = Path(__file__).with_name('2_cv.py').read_text(encoding='utf-8')
    schedule_class = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.ClassDef) and node.name == 'WarmUpCosine')
    namespace = {'tf': tf, 'np': np}
    exec(compile(ast.Module(body=[schedule_class], type_ignores=[]), str(__file__), 'exec'), namespace)
    schedule_type = namespace['WarmUpCosine']

    def create_model():
        # 与正式训练相同的 Keras SGD 和调度公式；小模型无 Dropout，便于比较下一步结果。
        with strategy.scope():
            schedule = schedule_type(0.01, 6, 0.001, 1)
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(2,)),
                tf.keras.layers.Dense(3, activation='softmax'),
            ])
            model.compile(optimizer=tf.keras.optimizers.SGD(schedule),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy())
        return model, schedule

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = {'protocol': 'checkpoint-verification', 'epochs': 3, 'steps_per_epoch': 1}
    x = np.ones((12, 2), dtype=np.float32)
    y = np.zeros((12,), dtype=np.int32)
    with tempfile.TemporaryDirectory(prefix='est-checkpoint-check-') as temporary:
        root = Path(temporary).resolve()
        original = root / 'original'
        resumed = root / 'resumed'
        original.mkdir()
        resumed.mkdir()
        model, schedule = create_model()
        callback = EstCheckpoints(model, config, original / 'best.h5')
        model.train_on_batch(x, y)
        callback.on_epoch_end(0, {'val_accuracy': 0.8})
        best_hash = digest(original / 'best.h5')
        best_predictions = model.predict_on_batch(x[:5])
        model.train_on_batch(x, y)
        callback.on_epoch_end(1, {'val_accuracy': 0.4})
        assert digest(original / 'best.h5') == best_hash
        assert len(list((original / 'latest').glob('epoch-*'))) == 1
        latest_weights = model.get_weights()

        # 模拟下一轮写到一半就中断：未发布目录及尚未提交的 best 均不能覆盖旧 latest。
        (original / 'latest' / '.pending-interrupted').mkdir()
        uncommitted = original / '.uncommitted.h5'
        other, _ = create_model()
        other.save_weights(str(uncommitted))
        os.replace(uncommitted, original / 'best.h5')
        restored, restored_schedule = create_model()
        recovered = EstCheckpoints(restored, config, resumed / 'best.h5', original)
        assert recovered.initial_epoch == 2
        assert int(restored.optimizer.iterations.numpy()) == 2
        assert int(recovered.best_epoch.numpy()) == 1
        assert float(recovered.best_score.numpy()) == 0.8
        assert digest(resumed / 'best.h5') == best_hash
        for expected, actual in zip(latest_weights, restored.get_weights()):
            np.testing.assert_allclose(expected, actual, rtol=0, atol=0)
        np.testing.assert_allclose(schedule(2).numpy(), restored_schedule(2).numpy(), rtol=0, atol=0)

        # 恢复后的下一次 SGD 更新须与未中断路径一致，证明优化器步数没有从零开始。
        model.train_on_batch(x, y)
        restored.train_on_batch(x, y)
        for expected, actual in zip(model.get_weights(), restored.get_weights()):
            np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7)
        recovered.on_epoch_end(2, {'val_accuracy': 0.6})
        assert digest(resumed / 'best.h5') == best_hash
        assert recovered.initial_epoch == 3
        assert int(restored.optimizer.iterations.numpy()) == 3
        restored.load_weights(str(resumed / 'best.h5'))
        np.testing.assert_allclose(best_predictions, restored.predict_on_batch(x[:5]),
                                   rtol=1e-6, atol=1e-7)

        # 配置不匹配必须直接报错，不能拿另一套实验的权重当作同一条续训。
        wrong = root / 'wrong'
        wrong.mkdir()
        mismatch, _ = create_model()
        try:
            EstCheckpoints(mismatch, dict(config, epochs=4), wrong / 'best.h5', original)
        except ValueError:
            pass
        else:
            raise AssertionError('修改训练配置后错误地允许恢复')
        pointer = json.loads((resumed / 'latest' / 'current.json').read_text())
        state = json.loads((resumed / 'latest' / pointer['directory'] / 'state.json').read_text())
        assert state['completed_epochs'] == 3 and state['best_epoch'] == 1
    print('EST_CHECKPOINT_ROUNDTRIP_OK', flush=True)


if __name__ == '__main__':
    # CPU 检查通过 Slurm 在计算节点运行；两张逻辑 CPU 覆盖镜像变量恢复路径。
    tf.config.set_visible_devices([], 'GPU')
    cpu = tf.config.list_physical_devices('CPU')[0]
    tf.config.set_logical_device_configuration(cpu, [
        tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()])
    strategy = tf.distribute.MirroredStrategy(devices=['/CPU:0', '/CPU:1'])
    verify_checkpoints(strategy)
