import argparse
import gc
from pathlib import Path

from common import (DEFAULT_CONFIG, environment, read_config, read_csv, read_json,
                    require_compute_node, sha256, verify_training, write_csv, write_json)
from data import load_fixed_splits
from analysis import summarize
from est import build_model, check_pretrained_cache, configure_gpus, make_dataset, predict_dataset

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='合并实验1：固定测试集上正确卫星与实际检索卫星的配对比较')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--splits', required=True)
    parser.add_argument('--est-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_compute_node()
    config = read_config(args.config)
    splits, manifest = load_fixed_splits(config, args.splits)
    checkpoint, training = verify_training(args.est_run, 'est', config, manifest)
    configure_gpus(config['est']['gpus'])
    check_pretrained_cache(config['est'])
    output = Path(args.output)
    if (output / 'predictions.csv').exists() or (output / 'result.json').exists():
        raise FileExistsError('该目录已包含Est评价结果')
    geo = read_json(output / 'geo.json')
    if geo['config'] != config or geo['split_sha256'] != manifest['split_sha256']:
        raise ValueError('Geo与Est的配置或数据划分不同')
    if sha256(output / 'geo_predictions.csv') != geo['predictions_sha256']:
        raise ValueError('Geo逐图检索结果发生变化')
    rows = read_csv(output / 'geo_predictions.csv')
    if [int(row['sample_id']) for row in rows] != [sample.sample_id for sample in splits.test]:
        raise ValueError('Geo结果必须覆盖固定测试集的全部查询，且顺序一致')
    gallery = {sample.satellite: sample for sample in splits.test}
    for sample, row in zip(splits.test, rows):
        if row['true_satellite'] != sample.satellite or int(row['true_label']) != sample.label:
            raise ValueError('Geo结果中的正确配对或查询标签不同')
        satellite = gallery[row['retrieved_satellite']]
        if (int(row['geo_correct']) != int(sample.sample_id == satellite.sample_id)
                or int(row['retrieved_satellite_label']) != satellite.label):
            raise ValueError('Geo检索图片的ID或类别记录不一致')
        sample.image_paths(config['data_root'])
    strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0', '/GPU:1'])
    with strategy.scope():
        model = build_model()
    model.load_weights(str(checkpoint))
    batch_size = config['est']['batch_size']
    vaData = make_dataset(splits.test, config['data_root'], batch_size)
    labels, paired = predict_dataset(model, vaData)
    del vaData
    gc.collect()
    retrievedData = make_dataset(splits.test, config['data_root'], batch_size,
                                  satellites=[row['retrieved_satellite'] for row in rows])
    combined_labels, combined = predict_dataset(model, retrievedData)
    np.testing.assert_array_equal(labels, [sample.label for sample in splits.test])
    np.testing.assert_array_equal(labels, combined_labels)
    if len(paired) != len(rows) or len(combined) != len(rows):
        raise ValueError('Est两种输入必须覆盖全部固定测试样本')
    original = read_csv(Path(args.est_run) / 'est_predictions.csv')
    if [row['svi_relative_path'] for row in original] != [row['svi_relative_path'] for row in rows]:
        raise ValueError('Est训练结束时的测试清单与串联评价清单不同')
    np.testing.assert_array_equal(paired.argmax(axis=1), [int(row['predicted_label']) for row in original])
    for index, row in enumerate(rows):
        row['paired_prediction'] = int(paired[index].argmax())
        row['combined_prediction'] = int(combined[index].argmax())
        if int(row['geo_correct']):
            np.testing.assert_allclose(paired[index], combined[index], atol=1e-6, rtol=1e-6)
        for label in range(3):
            row[f'paired_probability_{label}'] = float(paired[index, label])
            row[f'combined_probability_{label}'] = float(combined[index, label])
    result = summarize(rows, config['statistics'])
    result.update(experiment=config['experiment'], config=config,
                  split_sha256=manifest['split_sha256'],
                  geo_checkpoint_sha256=geo['checkpoint_sha256'],
                  est_checkpoint_sha256=training['checkpoint_sha256'],
                  environment=environment())
    write_csv(output / 'predictions.csv', rows)
    write_json(output / 'result.json', result)
    print('COMBINED_EXPERIMENT_1_COMPLETED', result['geo_incorrect'], flush=True)


if __name__ == '__main__':
    main()
