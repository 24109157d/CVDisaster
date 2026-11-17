import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import numpy as np
import tensorflow as tf


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), max=float(values.max()))


def probabilities(row, prefix):
    return np.array([float(row[f'{prefix}_probability_{i}']) for i in range(3)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-checks', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    source = args.run / 'source' / 'ME CVDisaster' / 'combined_experiment_1'
    sys.path.insert(0, str(source))
    from common import read_config, read_json, sha256
    from data import load_fixed_splits
    from est import build_model, load_image, make_dataset

    config = read_config(source / 'config.json')
    splits, manifest = load_fixed_splits(config, args.run / 'splits')
    training = read_json(args.run / 'est' / 'training.json')
    checkpoint = args.run / 'est' / training['checkpoint']
    assert sha256(checkpoint) == training['checkpoint_sha256']
    with (args.run / 'evaluation' / 'predictions.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row['sample_id']) for row in rows] == [sample.sample_id for sample in splits.test]
    wrong = [row for row in rows if row['geo_correct'] == '0']
    assert len(wrong) == 333
    root = Path(config['data_root'])
    input_checks = []
    if args.input_checks is not None:
        input_checks = read_json(args.input_checks)
        assert [item['sample_id'] for item in input_checks] == [row['sample_id'] for row in wrong]
        for item, row in zip(input_checks, wrong):
            assert item['true_satellite'] == str((root / row['true_satellite']).resolve(strict=True))
            assert item['retrieved_satellite'] == str((root / row['retrieved_satellite']).resolve(strict=True))
            assert sha256(Path(item['true_satellite'])) == item['paired_file_sha256']
            assert sha256(Path(item['retrieved_satellite'])) == item['retrieved_file_sha256']
            assert not item['tensor_equal']
        print('EXISTING_333_INPUT_CHECKS_AND_CURRENT_FILE_HASHES_VERIFIED', flush=True)
    for index, row in enumerate(wrong if args.input_checks is None else []):
        first = root / row['true_satellite']
        second = root / row['retrieved_satellite']
        a = load_image(str(first), 512).numpy()
        b = load_image(str(second), 512).numpy()
        item = dict(sample_id=row['sample_id'], true_satellite=str(first.resolve(strict=True)),
                    retrieved_satellite=str(second.resolve(strict=True)),
                    paired_file_sha256=sha256(first), retrieved_file_sha256=sha256(second),
                    tensor_shape=list(a.shape), pixel_mean_abs_difference=float(np.abs(a - b).mean()),
                    tensor_equal=bool(np.array_equal(a, b)))
        assert item['paired_file_sha256'] != item['retrieved_file_sha256']
        assert not item['tensor_equal']
        input_checks.append(item)
        if (index + 1) % 50 == 0:
            print('INPUTS_CHECKED', index + 1, flush=True)
    (args.output / 'input_checks.json').write_text(json.dumps(input_checks, indent=2))
    print('ALL_333_ACTUAL_SATELLITE_TENSORS_DIFFER', flush=True)

    # 使用全部逐图概率检查原第一类别与其他类别的差距。
    delta, margin, margin_change, ratio = [], [], [], []
    for row in wrong:
        a, b = probabilities(row, 'paired'), probabilities(row, 'combined')
        winner = int(a.argmax())
        others = [i for i in range(3) if i != winner]
        original_gap = a[winner] - a[others].max()
        possible_gap_change = max(abs((b[winner] - a[winner]) - (b[i] - a[i])) for i in others)
        assert b.argmax() == winner
        delta.append(float(np.abs(a - b).max()))
        margin.append(float(original_gap))
        margin_change.append(float(possible_gap_change))
        ratio.append(float(possible_gap_change / original_gap))
    result = dict(run=str(args.run), checkpoint_sha256=sha256(checkpoint),
                  diagnostic_source_sha256=sha256(Path(__file__)),
                  diagnostic_commit=os.environ.get('CVDISASTER_COMMIT'),
                  source_sha256={name: sha256(source / name) for name in ['est.py', 'estimate.py']},
                  split_sha256=manifest['split_sha256'], input_count=len(input_checks),
                  changed_tensor_count=sum(not item['tensor_equal'] for item in input_checks),
                  probability_max_abs_change=distribution(delta), original_class_margin=distribution(margin),
                  class_margin_change_bound=distribution(margin_change),
                  class_margin_change_over_original=distribution(ratio),
                  parameter_update_steps=0, inference_device='CPU', selected=[])
    selected_ids = []
    for row in sorted(wrong, key=lambda row: float(np.abs(probabilities(row, 'paired') - probabilities(row, 'combined')).max()), reverse=True)[:2]:
        selected_ids.append(row['sample_id'])
    for label in range(3):
        candidates = [row for row in wrong if int(row['true_label']) == label and row['retrieved_satellite_label'] != row['true_label']]
        selected_ids.append(min(candidates, key=lambda row: sorted(probabilities(row, 'paired'))[-1] - sorted(probabilities(row, 'paired'))[-2])['sample_id'])
    selected_ids = list(dict.fromkeys(selected_ids))
    print('BUILD_MODEL', selected_ids, flush=True)
    model = build_model()
    branch_initial = {name: model.get_layer(name).get_weights() for name in ['svi', 'sat']}
    model.load_weights(str(checkpoint))
    result['branches'] = {}
    for name in ['svi', 'sat']:
        branch = model.get_layer(name)
        before = branch_initial.pop(name)
        after = branch.get_weights()
        change = sum(float(np.sum((a.astype(np.float64) - b) ** 2)) for a, b in zip(after, before)) ** 0.5
        norm = sum(float(np.sum(a.astype(np.float64) ** 2)) for a in before) ** 0.5
        result['branches'][name] = dict(trainable=branch.trainable,
            trainable_parameter_count=sum(int(np.prod(v.shape)) for v in branch.trainable_weights),
            pretrained_parameter_change_l2=change, pretrained_parameter_change_relative=change / norm)
    concat = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Concatenate))
    dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert len(dense) == 2
    feature_model = tf.keras.Model(model.inputs, [*concat.input, model.output])
    kernel, bias = dense[0].get_weights()
    final_kernel, final_bias = dense[1].get_weights()
    feature_dim = int(concat.input[0].shape[-1])
    result['head'] = dict(input_width=int(kernel.shape[0]), svi_feature_width=feature_dim,
        sat_feature_width=int(concat.input[1].shape[-1]),
        svi_kernel_l2=float(np.linalg.norm(kernel[:feature_dim])),
        sat_kernel_l2=float(np.linalg.norm(kernel[feature_dim:])),
        sat_nonzero_weights=int(np.count_nonzero(kernel[feature_dim:])))
    (args.output / 'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False))

    for sample_id in selected_ids:
        row = next(row for row in wrong if row['sample_id'] == sample_id)
        sample = next(sample for sample in splits.test if str(sample.sample_id) == sample_id)
        ((ground_a, sat_a), _) = next(iter(make_dataset([sample], config['data_root'], 1)))
        ((ground_b, sat_b), _) = next(iter(make_dataset([sample], config['data_root'], 1, satellites=[row['retrieved_satellite']])))
        np.testing.assert_array_equal(ground_a, ground_b)
        np.testing.assert_array_equal(sat_a[0], load_image(str(root / row['true_satellite']), 512))
        np.testing.assert_array_equal(sat_b[0], load_image(str(root / row['retrieved_satellite']), 512))
        assert not np.array_equal(sat_a, sat_b)
        ground_feat, sat_feat_a, pred_a = [v.numpy() for v in feature_model((ground_a, sat_a), training=False)]
        ground_feat_b, sat_feat_b, pred_b = [v.numpy() for v in feature_model((ground_b, sat_b), training=False)]
        np.testing.assert_allclose(ground_feat, ground_feat_b, atol=1e-6, rtol=1e-6)
        assert int(pred_a.argmax()) == int(row['paired_prediction'])
        assert int(pred_b.argmax()) == int(row['combined_prediction'])
        svi_part = ground_feat @ kernel[:feature_dim]
        sat_part_a, sat_part_b = sat_feat_a @ kernel[feature_dim:], sat_feat_b @ kernel[feature_dim:]
        logits = []
        for part, expected in [(sat_part_a, pred_a), (sat_part_b, pred_b)]:
            hidden = dense[0].activation(tf.constant(svi_part + part + bias))
            logit = hidden.numpy() @ final_kernel + final_bias
            np.testing.assert_allclose(tf.nn.softmax(logit).numpy(), expected, atol=3e-6, rtol=3e-6)
            logits.append(logit)
        winner = int(pred_a.argmax())
        other = [i for i in range(3) if i != winner]
        item = dict(sample_id=sample_id, ground_tensor_equal=True, satellite_tensor_equal=False,
            saved_paired_probability=probabilities(row, 'paired').tolist(),
            saved_combined_probability=probabilities(row, 'combined').tolist(),
            replay_paired_probability=pred_a[0].tolist(), replay_combined_probability=pred_b[0].tolist(),
            cpu_vs_saved_gpu_paired_max_abs_difference=float(np.abs(pred_a[0] - probabilities(row, 'paired')).max()),
            cpu_vs_saved_gpu_combined_max_abs_difference=float(np.abs(pred_b[0] - probabilities(row, 'combined')).max()),
            svi_feature_l2=float(np.linalg.norm(ground_feat)),
            paired_sat_feature_l2=float(np.linalg.norm(sat_feat_a)),
            satellite_feature_change_l2=float(np.linalg.norm(sat_feat_a - sat_feat_b)),
            satellite_feature_change_relative=float(np.linalg.norm(sat_feat_a - sat_feat_b) / np.linalg.norm(sat_feat_a)),
            svi_hidden_contribution_l2=float(np.linalg.norm(svi_part)),
            paired_sat_hidden_contribution_l2=float(np.linalg.norm(sat_part_a)),
            satellite_hidden_change_l2=float(np.linalg.norm(sat_part_a - sat_part_b)),
            paired_logits=logits[0][0].tolist(), combined_logits=logits[1][0].tolist(),
            original_logit_margin=float(logits[0][0, winner] - logits[0][0, other].max()),
            max_logit_change=float(np.abs(logits[0] - logits[1]).max()))
        assert item['satellite_feature_change_l2'] > 0
        assert item['satellite_hidden_change_l2'] > 0
        result['selected'].append(item)
        (args.output / 'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print('REPLAY_VERIFIED', json.dumps(item), flush=True)
    assert sha256(checkpoint) == training['checkpoint_sha256']
    print('DIAGNOSIS_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
