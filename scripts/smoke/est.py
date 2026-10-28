"""读取三对真实图像，以原始分辨率执行 GCViT Tiny 单 batch 更新和独立样本推理。"""

import ast
from common import REPO, digest, execute, pairs


def work(data, report, protected):
    with report.stage('imports_and_gpu'):
        import numpy as np
        import tensorflow as tf
        from gcvit import GCViTTiny
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.random.set_seed(626)
        np.random.seed(626)
        gpus = tf.config.list_physical_devices('GPU')
        assert len(gpus) == 1, gpus
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_soft_device_placement(False)
        with tf.device('/GPU:0'):
            probe = tf.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
        assert float(tf.reduce_sum(probe).numpy()) == 4096
        report.state['gpu'] = {'framework': tf.__version__, 'device': probe.device,
                              'details': tf.config.experimental.get_device_details(gpus[0])}

    with report.stage('read_three_real_pairs'):
        samples = pairs(data, 3)
        street, satellite, labels = [], [], []
        for sample in samples:
            protected[sample['svi']] = digest(sample['svi'])
            protected[sample['sat']] = digest(sample['sat'])
            svi = tf.io.decode_png(tf.io.read_file(sample['svi']), channels=3)
            sat = tf.io.decode_png(tf.io.read_file(sample['sat']), channels=3)
            assert tuple(svi.shape) == (512, 1024, 3), svi.shape
            assert tuple(sat.shape) == (512, 512, 3), sat.shape
            # 与作者 ProcessPath 一致：resize 后保留 0..255 的 float32，不额外归一化。
            street.append(tf.image.resize(svi, (512, 1024)))
            satellite.append(tf.image.resize(sat, (512, 512)))
            labels.append(sample['label'])
        x0, x1 = tf.stack(street), tf.stack(satellite)
        y = tf.constant(labels, dtype=tf.int64)
        assert set(y.numpy().tolist()) == {0, 1, 2}
        assert bool(tf.reduce_all(tf.math.is_finite(x0)).numpy())
        assert bool(tf.reduce_all(tf.math.is_finite(x1)).numpy())
        report.state['samples'] = samples

    with report.stage('initialize_pretrained_dual_gcvit_tiny'):
        # 架构、输入尺寸、预训练加载及分类头均与作者 2_cv.py 对齐。
        left = GCViTTiny(input_shape=(512, 1024, 3), pretrain=True, resize_query=True)
        left.reset_classifier(num_classes=0, head_act=None)
        left._name = 'svi'
        right = GCViTTiny(input_shape=(512, 512, 3), pretrain=True, resize_query=True)
        right.reset_classifier(num_classes=0, head_act=None)
        right._name = 'sat'
        inputs = [tf.keras.Input((512, 1024, 3)), tf.keras.Input((512, 512, 3))]
        joined = tf.keras.layers.Concatenate()([left(inputs[0]), right(inputs[1])])
        head = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(joined)
        head = tf.keras.layers.Dropout(.2)(head)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(head)
        model = tf.keras.Model(inputs, outputs)
        assert all('GPU:0' in v.device for v in model.trainable_variables), '模型参数未放在 GPU'

        # 只提取学习率类，禁止导入会立即触发完整训练的 2_cv.py 顶层代码。
        source = REPO / 'Author CVDisaster/disaster_perception_mapping/2_cv.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WarmUpCosine')
        namespace = {'tf': tf, 'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        # 微型测试采用作者学习率曲线的首步 0.001；不模拟完整数据集的总训练步数。
        schedule = namespace['WarmUpCosine'](.01, 200, .001, 10)
        optimizer = tf.keras.optimizers.SGD(schedule)
        criterion = tf.keras.losses.SparseCategoricalCrossentropy()

    with report.stage('one_training_batch'):
        # eager GradientTape 只执行一次；不调用 fit，不缓存全量数据，也不保存模型。
        tracked = left.trainable_variables[0]
        before = tracked.numpy().copy()
        with tf.GradientTape() as tape:
            probabilities = model([x0[:1], x1[:1]], training=True)
            loss = criterion(y[:1], probabilities)
        assert 'GPU:0' in probabilities.device, '训练前向输出未位于 GPU'
        assert np.isfinite(loss.numpy()) and loss.numpy() > 0
        gradients = tape.gradient(loss, model.trainable_variables)
        # 相对位置偏置的 gather 可产生 IndexedSlices；有限性检查兼容稀疏梯度。
        assert all(g is not None for g in gradients), '存在未连接到 loss 的可训练参数'
        dense_gradients = [tf.convert_to_tensor(g) for g in gradients]
        assert all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in dense_gradients)
        assert any(bool(tf.reduce_any(g != 0).numpy()) for g in dense_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        assert int(optimizer.iterations.numpy()) == 1
        assert not np.array_equal(before, tracked.numpy()), '优化器未实际更新骨干参数'
        report.state['training'] = {'batches': 1, 'batch_size': 1, 'loss': float(loss.numpy()),
                                    'labels': y[:1].numpy().tolist(), 'parameters_updated': True,
                                    'parameter_device': tracked.device, 'output_device': probabilities.device}
        del gradients, dense_gradients, before

    with report.stage('minimal_heldout_inference'):
        probabilities = model([x0[1:], x1[1:]], training=False).numpy()
        assert probabilities.shape == (2, 3) and np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        report.state['inference'] = {'count': 2, 'labels': y[1:].numpy().tolist(),
            'predictions': probabilities.argmax(axis=1).tolist(), 'probabilities': probabilities.tolist(),
            'meaning': '只验证推理通路，不计算正式分类准确率'}
        report.state['peak_gpu_allocated_mib'] = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2**20


if __name__ == '__main__':
    execute('est', work)
