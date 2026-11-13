import argparse
from pathlib import Path, PurePosixPath

from common import (DEFAULT_CONFIG, environment, new_directory, read_config, require_compute_node,
                    sha256, verify_training, write_csv, write_json)
from data import load_fixed_splits
from geo import create_model, make_loaders, require_gpus, runtime_config

import torch
from sample4geo.evaluate.cvusa_and_cvact import calculate_scores
from sample4geo.trainer import predict


def main():
    parser = argparse.ArgumentParser(description='合并实验1：完整固定测试集的Geo Top-1检索')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--splits', required=True)
    parser.add_argument('--geo-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_compute_node()
    config = read_config(args.config)
    require_gpus(config['geo']['gpus'])
    splits, manifest = load_fixed_splits(config, args.splits)
    checkpoint, training = verify_training(args.geo_run, 'geo', config, manifest)
    output = new_directory(args.output)
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint), strict=False)
    model = torch.nn.DataParallel(model, device_ids=list(range(config['geo']['gpus'])))
    model = model.to('cuda')
    datasets, loaders = make_loaders(config, model.module)
    expected_ids = [sample.sample_id for sample in splits.test]
    for view in ['query', 'reference']:
        if list(datasets['test_' + view].label) != expected_ids:
            raise ValueError('Geo测试顺序与固定清单不同')
    runtime = runtime_config(config)
    reference_features, reference_ids = predict(runtime, model, loaders['test_reference'])
    query_features, query_ids = predict(runtime, model, loaders['test_query'])
    if not torch.isfinite(reference_features).all() or not torch.isfinite(query_features).all():
        raise FloatingPointError('Geo特征包含非有限数值')
    details = calculate_scores(query_features, reference_features, query_ids, reference_ids,
                               return_details=True)
    scores = details['scores']
    top_indices = scores.argmax(dim=1).tolist()
    rows = []
    for index, top_index in enumerate(top_indices):
        sample = splits.test[index]
        satellite = splits.test[top_index]
        rows.append({'sample_id': sample.sample_id,
                     'svi_relative_path': str(PurePosixPath(*PurePosixPath(sample.raw_ground).parts[1:])),
                     'true_label': sample.label, 'true_satellite': sample.satellite,
                     'retrieved_satellite': satellite.satellite,
                     'retrieved_satellite_label': satellite.label,
                     'geo_correct': int(sample.sample_id == satellite.sample_id),
                     'similarity': float(scores[index, top_index]),
                     'top1_tie_count': int((scores[index] == scores[index, top_index]).sum())})
    write_csv(output / 'geo_predictions.csv', rows)
    write_json(output / 'geo.json', {
        'config': config, 'split_sha256': manifest['split_sha256'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': training['checkpoint_sha256'],
        'predictions_sha256': sha256(output / 'geo_predictions.csv'),
        'metrics': details['metrics'], 'query_count': len(splits.test), 'gallery_count': len(splits.test),
        'top1_accuracy': sum(row['geo_correct'] for row in rows) / len(rows),
        'tie_rule': 'first_gallery_index', 'environment': environment()})


if __name__ == '__main__':
    main()
