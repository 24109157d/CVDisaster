import argparse
import math
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from common import (check_manifest_settings, configuration, image_path, load_manifest, new_directory,
                    read_json, sha256, write_json)
from dataset import make_dataset, noise_image
from evaluate import parameter_fingerprint
from models import ClassificationLoss, VARIANTS, build_model, compile_model, policy
from training import AccumulatingModel, Checkpoints, configure_devices


def real_inputs(cases, settings, variant, index=0):
    p = policy(variant)
    target = cases['targets'][index]

    def load(relative, shape):
        path = image_path(cases['data_root'], relative)
        return tf.image.resize(tf.io.decode_png(tf.io.read_file(path), channels=3), shape[:2])

    result = {}
    if p['ground']:
        result['ground'] = load(target['ground'], settings['ground_shape'])[None]
    if p['satellite']:
        result['satellite'] = load(target['satellite'], settings['satellite_shape'])[None]
    if p['neighbors']:
        entries = cases['neighbors'][target['sample_id']]
        if len(entries) != settings['neighbors']:
            raise ValueError('真实验证样本的邻图数量不符合配置')
        images, relations = [], []
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        for sample_id in entries:
            source = cases['samples'][sample_id]
            _, _, distance = geod.inv(float(target['longitude']), float(target['latitude']),
                                      float(source['longitude']), float(source['latitude']))
            delta = float(source['timestamp']) - float(target['timestamp'])
            images.append(load(source['ground'], settings['ground_shape']))
            relations.append([distance / settings['radius_m'], delta / settings['time_seconds']])
        result['neighbors'] = tf.stack(images)[None]
        result['mask'] = tf.ones([1, settings['neighbors']])
        result['relations'] = tf.constant([relations], dtype=tf.float32)
    return result, int(target['label'])


def main():
    parser = argparse.ArgumentParser(description='真实 CVIAN 图片及 ImageNet 权重的模型验证')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    cases = read_json(args.cases)
    settings = configuration()
    check_manifest_settings(settings, load_manifest(cases['manifest']))
    if sha256(Path(cases['manifest']) / 'manifest.json') != cases['manifest_sha256']:
        raise ValueError('真实验证样本与固定清单指纹不一致')
    settings['initial_weights'] = cases['weights']
    settings['data_root'] = cases['data_root']
    if sha256(cases['weights']) != settings['initial_sha256']:
        raise ValueError('真实预训练权重指纹不一致')
    strategy = configure_devices(args.gpus)
    tf.keras.utils.set_random_seed(settings['seed'])
    with strategy.scope():
        functional = build_model(settings, args.variant)
        model = AccumulatingModel(functional, samples=len(cases['targets']))
        compile_model(model, settings, steps=1)
    x, label = real_inputs(cases, settings, args.variant)
    probabilities = model(x, training=False)['final'].numpy()
    assert probabilities.shape == (1, 3)
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-6)
    assert np.isfinite(probabilities).all()
    gate = model.get_layer('uncertainty_gate')
    initial_prediction = model(x, training=False)
    np.testing.assert_allclose(initial_prediction['final'], initial_prediction['base'], atol=1e-7)
    gate.progress.assign(settings['gate_full_epoch'])
    probabilities = model(x, training=False)['final'].numpy()
    with tf.GradientTape() as tape:
        base_probability = tf.Variable(initial_prediction['base'])
        tape.watch(base_probability)
        routed = gate(base_probability)
    assert tape.gradient(routed, base_probability) is None
    with tf.GradientTape() as tape:
        active_prediction = model(x, training=False)
        active_loss = sum(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            tf.constant([label]), value)) for value in active_prediction.values())
    gradients = tape.gradient(active_loss, model.trainable_variables)
    assert all(value is not None and bool(tf.reduce_all(tf.math.is_finite(value))) for value in gradients)
    checks = ['real_image_forward', 'warmup_base_identity', 'entropy_stop_gradient']
    checks.append('active_gate_gradients')
    if policy(args.variant)['neighbors']:
        order = list(reversed(range(settings['neighbors'])))
        permuted = {**x, 'neighbors': tf.gather(x['neighbors'], order, axis=1),
                    'mask': tf.gather(x['mask'], order, axis=1),
                    'relations': tf.gather(x['relations'], order, axis=1)}
        np.testing.assert_allclose(model(permuted, training=False)['final'], probabilities, atol=2e-5)
        empty = {**x, 'mask': tf.zeros_like(x['mask'])}
        other = {**empty, 'neighbors': tf.repeat(x['ground'][:, None], settings['neighbors'], axis=1)}
        np.testing.assert_allclose(model(empty, training=False)['final'],
                                   model(other, training=False)['final'], atol=1e-6)
        empty_prediction = model(empty, training=False)
        np.testing.assert_allclose(empty_prediction['final'], empty_prediction['base'], atol=1e-6)
        checks += ['neighbor_permutation', 'empty_evidence']
        if settings['neighbors'] == 2:
            duplicates = {**x, 'neighbors': tf.repeat(x['neighbors'][:, :1], settings['neighbors'], axis=1),
                          'relations': tf.repeat(x['relations'][:, :1], settings['neighbors'], axis=1)}
            single = {**duplicates, 'mask': tf.constant([[1.0, 0.0]])}
            np.testing.assert_allclose(model(duplicates, training=False)['final'],
                                       model(single, training=False)['final'], atol=2e-5)
            checks.append('duplicate_evidence_mass')
    actual_probability = model(x, training=False)['final']
    criterion = ClassificationLoss(settings['lambda_ordinal'], settings['ordinal_boundary_weights'])
    observed_loss = float(criterion(tf.constant([label]), actual_probability))
    raw = actual_probability.numpy()[0]
    expected_loss = -np.log(raw[label]) + settings['lambda_ordinal'] * 0.5 * sum(
        settings['ordinal_boundary_weights'][k] * (raw[:k + 1].sum() - float(label <= k)) ** 2
        for k in range(2))
    np.testing.assert_allclose(observed_loss, expected_loss, atol=1e-6)
    checks.append('ordinal_loss_real_prediction')
    if policy(args.variant)['satellite']:
        other_x, _ = real_inputs(cases, settings, args.variant, index=1)
        wrong = {**x, 'satellite': other_x['satellite']}
        assert not np.array_equal(x['satellite'], wrong['satellite'])
        noisy = {**x, 'satellite': tf.constant(noise_image(cases['targets'][0]['sample_id'], 42,
                                                          settings['satellite_shape']))[None]}
        fingerprint = parameter_fingerprint(model)
        for condition in (wrong, noisy):
            result = model(condition, training=False)['final'].numpy()
            assert np.isfinite(result).all()
        assert parameter_fingerprint(model) == fingerprint
        checks.append('satellite_replacement_without_updates')
    before = {name: [v.numpy().copy() for v in model.get_layer(name).weights]
              for name in ('svi', 'sat') if name in [layer.name for layer in model.layers]}
    actual = [real_inputs(cases, settings, args.variant, index=i) for i in range(len(cases['targets']))]
    batch_x = {key: tf.concat([item[0][key] for item in actual], axis=0) for key in x}
    batch_y = {name: tf.constant([item[1] for item in actual], dtype=tf.int64) for name in model.output}
    training_metrics = model.train_on_batch(batch_x, batch_y, return_dict=True)
    assert all(math.isfinite(value) for value in training_metrics.values())
    assert int(model.optimizer.iterations.numpy()) == 1
    if args.gpus:
        for variable in model.trainable_variables:
            replicas = strategy.experimental_local_results(variable)
            assert len(replicas) == args.gpus
            for replica in replicas[1:]:
                np.testing.assert_allclose(replicas[0].numpy(), replica.numpy(), rtol=1e-6, atol=1e-7)
        checks.append('mirrored_parameter_updates')
    for name, original in before.items():
        assert any(not np.array_equal(a, b.numpy()) for a, b in zip(original, model.get_layer(name).weights))
    checks.append('real_image_backward_and_branch_updates')
    output = new_directory(args.output)
    config = {'optimizer_steps_per_epoch': 1, 'stop_epoch': 1, 'variant': args.variant,
              'settings': settings, 'epochs': 200}
    callback = Checkpoints(model, output, config, selection=True)
    metric_name = 'final_accuracy' if len(model.output) > 1 else 'accuracy'
    score = float(training_metrics[metric_name])
    callback.on_epoch_end(0, {f'val_{metric_name}': score, **training_metrics})
    expected = parameter_fingerprint(model)
    started = time.monotonic()
    model.train_on_batch(batch_x, batch_y)
    step_seconds = time.monotonic() - started
    assert int(model.optimizer.iterations.numpy()) == 2
    restored = new_directory(output / 'restored')
    restored_callback = Checkpoints(model, restored, config, selection=True, resume=output)
    assert restored_callback.completed == 1
    assert int(model.optimizer.iterations.numpy()) == 1
    assert parameter_fingerprint(model) == expected
    checks.append('checkpoint_weights_optimizer_roundtrip')
    before_evaluation = parameter_fingerprint(model)
    for condition in ('paired', 'wrong', 'noise', 'gaussian', 'zero'):
        dataset, rows = make_dataset(cases['manifest'], 'validation', settings, args.variant,
                                     settings['batch_size'], condition=condition)
        batch, labels = next(iter(dataset))
        assert set(batch) == set(model.input)
        assert set(labels) == set(model.output)
        np.testing.assert_array_equal(labels['final'], [int(row['label']) for row in rows[:settings['batch_size']]])
        result = model.predict_on_batch(batch)
        for probabilities in result.values():
            assert probabilities.shape == (settings['batch_size'], 3)
            assert np.isfinite(probabilities).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert parameter_fingerprint(model) == before_evaluation
    checks.append('real_dataloader_all_satellite_conditions')
    write_json(output / 'verification.json', {'variant': args.variant, 'checks': checks,
               'image_shapes': {'ground': settings['ground_shape'], 'satellite': settings['satellite_shape']},
               'training_metrics': {key: float(value) for key, value in training_metrics.items()},
               'training_batch_size': len(actual), 'steady_step_seconds': step_seconds,
               'parameters': model.count_params(), 'pretrained_sha256': settings['initial_sha256'],
               'tensorflow': tf.__version__, 'gpus': args.gpus})
    print('REAL_MODEL_VERIFIED', args.variant, checks, flush=True)


if __name__ == '__main__':
    main()
