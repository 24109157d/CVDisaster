import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import tensorflow as tf

from common import (check_manifest_settings, environment, load_manifest, new_directory, read_csv, read_json,
                    model_fingerprints, require_compute_node, sha256, write_csv, write_json)
from dataset import make_dataset, noise_image, read_split, wrong_permutation
from conditions import noise_fingerprint
from conditions import gaussian_image
from analysis import net_benefit
from models import build_model, policy
from training import AccumulatingModel, configure_devices


def metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(labels), 3) or not np.isfinite(probabilities).all():
        raise ValueError('分类概率形状或数值无效')
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('分类概率超出 [0,1]')
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels, predicted, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return {'count': len(labels), 'accuracy': float(accuracy_score(labels, predicted)),
            'macro_f1': report['macro avg']['f1-score'], 'report': report,
            'confusion_matrix': confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist(),
            'nll': float(log_loss(labels, probabilities, labels=[0, 1, 2]))}


def predict(model, dataset):
    probabilities = []
    base_probabilities = []
    labels = []
    for x, y in dataset:
        prediction = model.predict_on_batch(x)
        probabilities.append(prediction['final'])
        base_probabilities.append(prediction['base'])
        labels.append(y['final'].numpy())
    return np.concatenate(labels), np.concatenate(probabilities), np.concatenate(base_probabilities)


def parameter_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for variable in model.weights:
        value = variable.numpy()
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='固定 EST 权重的卫星替换评价')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geo-predictions', type=Path)
    parser.add_argument('--gpus', type=int, choices=(0, 2), default=2)
    args = parser.parse_args()
    require_compute_node()
    manifest = load_manifest(args.manifest)
    run = read_json(args.run / 'training.json')
    if run['stage'] != 'refit' or run['completed_epochs'] != run['stop_epoch']:
        raise ValueError('正式测试只能使用已完成的完整训练集合重训产物')
    if run['manifest_sha256'] != sha256(args.manifest / 'manifest.json'):
        raise ValueError('评价与训练的固定划分不一致')
    if run['model_source_sha256'] != model_fingerprints():
        raise ValueError('评价与训练的模型或数据代码不一致')
    checkpoint = args.run / 'selected.h5'
    if sha256(checkpoint) != run['selected_sha256']:
        raise ValueError('选定模型权重指纹不一致')
    settings = run['settings']
    check_manifest_settings(settings, manifest)
    strategy = configure_devices(args.gpus)
    with strategy.scope():
        functional = build_model(settings, run['variant'])
        model = AccumulatingModel(functional, run['counts']['refit']['targets'], settings['batch_size'])
        model.load_weights(str(checkpoint))
    before = parameter_fingerprint(model)
    output = new_directory(args.output)
    write_json(output / 'environment.json', environment())
    target_rows, _, _ = read_split(args.manifest, 'test')
    geo_paths = None
    geo_wrong = None
    if args.geo_predictions is not None:
        source = read_csv(args.geo_predictions)
        geo_paths = {item['sample_id']: item['retrieved_satellite'] for item in source}
        if len(geo_paths) != len(source):
            raise ValueError('Geo 检索记录出现重复 ID')
        expected = {item['sample_id']: item for item in target_rows}
        if set(geo_paths) != set(expected):
            raise ValueError('Geo 检索目标与当前测试清单不一致')
        for item in source:
            if (item['true_satellite'] != expected[item['sample_id']]['satellite']
                    or int(item['true_label']) != int(expected[item['sample_id']]['label'])):
                raise ValueError('Geo 检索配对或标签与测试清单不一致')
        geo_wrong = np.asarray([geo_paths[item['sample_id']] != item['satellite'] for item in target_rows])
    conditions = [('paired', 42), ('zero', 42)]
    conditions += [(name, seed) for name in ('wrong', 'noise', 'gaussian') for seed in range(42, 47)]
    if geo_paths is not None:
        conditions.append(('geo', 42))
    records, summary = [], {}
    started = time.monotonic()
    for condition, seed in conditions:
        dataset, rows = make_dataset(args.manifest, 'test', settings, run['variant'],
                                     settings['batch_size'], condition, seed, geo_paths,
                                     random_neighbors=run['random_neighbors'])
        labels, probabilities, base_probabilities = predict(model, dataset)
        if len(labels) != len(target_rows):
            raise ValueError('正式测试未覆盖全部目标')
        expected_labels = [int(row['label']) for row in rows]
        np.testing.assert_array_equal(labels, expected_labels)
        key = f'{condition}_{seed}'
        summary[key] = metrics(labels, probabilities)
        summary[key]['base'] = metrics(labels, base_probabilities)
        summary[key]['neighbor_net_benefit'] = net_benefit(
            base_probabilities.argmax(axis=1) == labels, probabilities.argmax(axis=1) == labels)
        summary[key]['neighbor_net_benefit_by_class'] = {
            str(c): net_benefit(base_probabilities[labels == c].argmax(axis=1) == c,
                               probabilities[labels == c].argmax(axis=1) == c)
            for c in range(3) if np.any(labels == c)}
        if geo_wrong is not None and geo_wrong.any():
            summary[key]['geo_wrong_subset'] = metrics(labels[geo_wrong], probabilities[geo_wrong])
        permutation = wrong_permutation(len(rows), seed) if condition == 'wrong' else np.arange(len(rows))
        for index, row in enumerate(rows):
            satellite = geo_paths[row['sample_id']] if condition == 'geo' else rows[permutation[index]]['satellite']
            noise_hash = noise_fingerprint(row['sample_id'], seed, settings['satellite_shape']) if condition == 'noise' else ''
            if condition == 'gaussian':
                noise_hash = hashlib.sha256(gaussian_image(
                    row['sample_id'], seed, settings['satellite_shape']).astype('<f4').tobytes()).hexdigest()
            records.append({'sample_id': row['sample_id'], 'label': int(labels[index]),
                            'sequence': row['sequence'], 'spatial_group': row.get('spatial_group', ''),
                            'condition': condition, 'condition_seed': seed, 'training_seed': run['seed'],
                            'variant': run['variant'], 'satellite': '' if condition in ('noise', 'gaussian', 'zero') else satellite,
                            'noise_sha256': noise_hash,
                            'geo_wrong': int(geo_wrong[index]) if geo_wrong is not None else '',
                            'prediction': int(probabilities[index].argmax()),
                            'base_prediction': int(base_probabilities[index].argmax()),
                            **{f'base_p{c}': float(base_probabilities[index, c]) for c in range(3)},
                            **{f'p{c}': float(probabilities[index, c]) for c in range(3)}})
        print('EVALUATED', key, summary[key]['accuracy'], flush=True)
    if parameter_fingerprint(model) != before or sha256(checkpoint) != run['selected_sha256']:
        raise ValueError('评价过程中模型参数发生改变')
    write_csv(output / 'predictions.csv', records)
    write_json(output / 'result.json', {'run': run, 'metrics': summary,
               'parameter_updates': 0, 'checkpoint_sha256': run['selected_sha256'],
               'manifest_sha256': run['manifest_sha256'], 'elapsed_seconds': time.monotonic() - started,
               'geo_predictions_sha256': sha256(args.geo_predictions) if args.geo_predictions else None})


if __name__ == '__main__':
    main()
