import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np


def make_noise(seed, sample_id):
    generator = np.random.default_rng(np.random.SeedSequence([seed, sample_id]))
    return generator.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)


def weight_fingerprint(weights):
    digest = hashlib.sha256()
    for weight in weights:
        value = np.ascontiguousarray(weight.numpy())
        digest.update(weight.name.encode('utf-8'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='使用已有Est best比较配对卫星、Geo检索卫星与随机噪声卫星输入')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError('噪声seed必须为非负整数')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.run / 'source' / 'ME CVDisaster' / 'combined_experiment_1'
    sys.path.insert(0, str(source))
    from common import (CLASS_NAMES, environment, read_config, read_csv, require_compute_node,
                        sha256, verify_training, write_csv, write_json)
    from data import load_fixed_splits
    from analysis import classification_metrics, paired_effect
    from est import build_model, load_image
    import tensorflow as tf

    require_compute_node()
    if tf.config.list_physical_devices('GPU'):
        raise RuntimeError('本次诊断只允许CPU推理')
    config = read_config(source / 'config.json')
    splits, manifest = load_fixed_splits(config, args.run / 'splits')
    checkpoint, training = verify_training(args.run / 'est', 'est', config, manifest)
    source_rows = read_csv(args.run / 'evaluation' / 'predictions.csv')
    samples = splits.test
    count = len(samples)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    np.testing.assert_array_equal([int(row['sample_id']) for row in source_rows],
                                  [sample.sample_id for sample in samples])
    gallery = {sample.satellite: index for index, sample in enumerate(samples)}
    retrieved_indices = np.array([gallery[row['retrieved_satellite']] for row in source_rows])
    geo_wrong = retrieved_indices != np.arange(count)
    np.testing.assert_array_equal(geo_wrong, [row['geo_correct'] == '0' for row in source_rows])
    for sample, row in zip(samples, source_rows):
        if row['true_satellite'] != sample.satellite or int(row['true_label']) != sample.label:
            raise ValueError('原始逐图预测中的配对图片或标签与固定清单不一致')

    model = build_model()
    model.load_weights(str(checkpoint))
    initial_fingerprint = weight_fingerprint(model.weights)
    concat = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Concatenate))
    dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dropout = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    if len(dense) != 2 or len(dropout) != 1:
        raise ValueError('Est分类层结构与原模型不一致')
    feature_model = tf.keras.Model(model.inputs, [*concat.input, model.output])
    paired_forward = tf.function(lambda a, b: feature_model((a, b), training=False),
        input_signature=[tf.TensorSpec((1, 512, 1024, 3), tf.float32),
                         tf.TensorSpec((1, 512, 512, 3), tf.float32)])
    noise_forward = tf.function(lambda image: model.get_layer('sat')(image, training=False),
        input_signature=[tf.TensorSpec((1, 512, 512, 3), tf.float32)])

    def classify(ground_feature, satellite_feature):
        hidden = dense[0](concat([ground_feature, satellite_feature]), training=False)
        hidden = dropout[0](hidden, training=False)
        return dense[1](hidden, training=False)

    head_forward = tf.function(classify, input_signature=[
        tf.TensorSpec((None, 512), tf.float32), tf.TensorSpec((None, 512), tf.float32)])
    root = Path(config['data_root'])
    ground_features, satellite_features, noise_features, actual_paired = [], [], [], []
    noise_records, direct_checks = [], []
    checks = {0, count // 2, count - 1}
    print('CPU_NOISE_INFERENCE_STARTED', count, 'seed=', args.seed, flush=True)
    for index, sample in enumerate(samples):
        ground = load_image(str(root / sample.raw_ground), 1024)[None]
        satellite = load_image(str(root / sample.satellite), 512)[None]
        noise = make_noise(args.seed, sample.sample_id)
        noise_tensor = tf.cast(tf.convert_to_tensor(noise[None]), tf.float32)
        ground_feature, satellite_feature, paired = paired_forward(ground, satellite)
        noise_feature = noise_forward(noise_tensor)
        ground_features.append(ground_feature.numpy())
        satellite_features.append(satellite_feature.numpy())
        noise_features.append(noise_feature.numpy())
        actual_paired.append(paired.numpy())
        noise_records.append(dict(noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                                  noise_mean=float(noise.mean()), noise_std=float(noise.std())))
        if index in checks:
            direct_noise = model((ground, noise_tensor), training=False).numpy()
            reconstructed_noise = head_forward(ground_feature, noise_feature).numpy()
            np.testing.assert_allclose(direct_noise, reconstructed_noise, atol=1e-5, rtol=1e-5)
            wrong_satellite = load_image(str(root / samples[retrieved_indices[index]].satellite), 512)[None]
            direct_retrieved = model((ground, wrong_satellite), training=False).numpy()
            direct_checks.append(dict(index=index, sample_id=sample.sample_id,
                noise_full_model=direct_noise.tolist(), retrieved_full_model=direct_retrieved.tolist(),
                noise_head_max_difference=float(np.abs(direct_noise - reconstructed_noise).max())))
        if (index + 1) % 25 == 0 or index + 1 == count:
            print('CPU_NOISE_IMAGES_PROCESSED', index + 1, '/', count, flush=True)

    ground_features, satellite_features, noise_features, paired = [np.concatenate(values) for values in
        [ground_features, satellite_features, noise_features, actual_paired]]
    paired_from_head = head_forward(ground_features, satellite_features).numpy()
    retrieved = head_forward(ground_features, satellite_features[retrieved_indices]).numpy()
    noise = head_forward(ground_features, noise_features).numpy()
    np.testing.assert_allclose(paired, paired_from_head, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(paired.argmax(axis=1), paired_from_head.argmax(axis=1))
    for check in direct_checks:
        np.testing.assert_allclose(retrieved[check['index']], check['retrieved_full_model'][0], atol=1e-5, rtol=1e-5)
    for probability in [paired, retrieved, noise]:
        if probability.shape != (count, 3) or not np.isfinite(probability).all():
            raise ValueError('推理概率形状或数值不符合要求')
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(paired[~geo_wrong], retrieved[~geo_wrong], atol=1e-5, rtol=1e-5)
    probabilities = dict(paired_satellite=paired, geo_retrieved_satellite=retrieved, uniform_noise_satellite=noise)

    def comparison(mask):
        result = {}
        for name, probability in probabilities.items():
            prediction = probability[mask].argmax(axis=1)
            metrics = classification_metrics(labels[mask], prediction)
            metrics['predicted_class_counts'] = np.bincount(prediction, minlength=3).tolist()
            metrics['mean_class_probabilities'] = probability[mask].mean(axis=0).tolist()
            result[name] = metrics
        result['geo_vs_paired'] = paired_effect(labels[mask], paired[mask].argmax(axis=1),
                                               retrieved[mask].argmax(axis=1), config['statistics'])
        result['noise_vs_paired'] = paired_effect(labels[mask], paired[mask].argmax(axis=1),
                                                 noise[mask].argmax(axis=1), config['statistics'])
        for name in ['geo_retrieved_satellite', 'uniform_noise_satellite']:
            difference = 100 * np.abs(probabilities[name][mask] - paired[mask])
            result[name]['changed_prediction_count'] = int((probabilities[name][mask].argmax(axis=1) != paired[mask].argmax(axis=1)).sum())
            result[name]['mean_abs_class_probability_change_percentage_points'] = difference.mean(axis=0).tolist()
            result[name]['max_abs_probability_change_percentage_points'] = float(difference.max())
        return result

    output_rows = []
    for index, (sample, source_row) in enumerate(zip(samples, source_rows)):
        row = dict(sample_id=sample.sample_id, true_label=sample.label, ground=sample.raw_ground,
                   true_satellite=sample.satellite, retrieved_satellite=source_row['retrieved_satellite'],
                   geo_correct=int(not geo_wrong[index]), noise_seed=args.seed, **noise_records[index])
        for name, probability in probabilities.items():
            row[name + '_prediction'] = int(probability[index].argmax())
            for label in range(3):
                row[f'{name}_probability_{label}'] = float(probability[index, label])
        output_rows.append(row)
    saved_paired = np.array([[float(row[f'paired_probability_{label}']) for label in range(3)] for row in source_rows])
    saved_retrieved = np.array([[float(row[f'combined_probability_{label}']) for label in range(3)] for row in source_rows])
    result = dict(experiment='satellite_uniform_noise_inference', reference_run=str(args.run),
        checkpoint=str(checkpoint), checkpoint_sha256=training['checkpoint_sha256'],
        best_epoch=training['best_epoch'], split_sha256=manifest['split_sha256'], class_names=CLASS_NAMES,
        inference_device='CPU', parameter_update_steps=0, inference_batch_size=1,
        noise=dict(distribution='各像素与RGB通道独立的离散均匀分布', seed=args.seed,
            per_query_seed='numpy.SeedSequence([seed, sample_id])', rng='PCG64',
            shape=[512, 512, 3], integer_range_inclusive=[0, 255], input_dtype='float32',
            image_files_written=0),
        full_test=comparison(np.ones(count, dtype=bool)), geo_incorrect_only=comparison(geo_wrong),
        cpu_vs_archived_gpu=dict(paired_max_abs_probability_difference=float(np.abs(paired - saved_paired).max()),
            paired_changed_prediction_count=int((paired.argmax(axis=1) != saved_paired.argmax(axis=1)).sum()),
            retrieved_max_abs_probability_difference=float(np.abs(retrieved - saved_retrieved).max()),
            retrieved_changed_prediction_count=int((retrieved.argmax(axis=1) != saved_retrieved.argmax(axis=1)).sum())),
        full_model_checks=direct_checks, environment=environment(),
        diagnostic_source_sha256=sha256(Path(__file__)))
    if weight_fingerprint(model.weights) != initial_fingerprint or sha256(checkpoint) != training['checkpoint_sha256']:
        raise RuntimeError('推理前后模型权重发生变化')
    result['model_weight_fingerprint_before_and_after'] = initial_fingerprint
    write_csv(args.output / 'predictions.csv', output_rows)
    np.savez_compressed(args.output / 'features.npz', sample_ids=np.array([sample.sample_id for sample in samples]),
                        ground_features=ground_features, satellite_features=satellite_features, noise_features=noise_features)
    result['predictions_sha256'] = sha256(args.output / 'predictions.csv')
    write_json(args.output / 'result.json', result)
    print('CPU_NOISE_INFERENCE_COMPLETED', {name: values['accuracy'] for name, values in result['full_test'].items()
          if name in probabilities}, flush=True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
