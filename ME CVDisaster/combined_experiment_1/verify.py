import argparse
import ast
import csv
import math
from pathlib import Path, PurePosixPath
import uuid

import numpy as np

from analysis import classification_metrics, paired_effect
from common import (CLASS_NAMES, DEFAULT_CONFIG, REPOSITORY, read_config, read_csv, read_json,
                    sample_ids_for_record, write_json)
from data import read_samples
from prepare_splits import expected_pairs


def check_est_structure():
    author = ast.parse((REPOSITORY / 'Author CVDisaster' / 'disaster_perception_mapping' / '2_cv.py').read_text(encoding='utf-8'))
    combined = ast.parse(Path(__file__).with_name('est.py').read_text(encoding='utf-8'))
    build = next(node for node in combined.body if isinstance(node, ast.FunctionDef) and node.name == 'build_model')
    names = ['featSVI', 'featSAT', 'concat', 'output']

    def assignments(nodes):
        return [ast.dump(node, include_attributes=False) for node in nodes
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names]

    scope = next(node for node in author.body if isinstance(node, ast.With)
                 and any(isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                         and child.targets[0].id == 'model' for child in node.body))
    assert assignments(scope.body) == assignments(build.body)
    constants = {node.targets[0].id: node.value.value for node in author.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, ast.Constant)}
    for name in ['inpSVI', 'inpSAT']:
        first = next(node.value for node in scope.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
        second = next(node.value for node in build.body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
        assert [constants[item.id] if isinstance(item, ast.Name) else item.value
                for item in first.keywords[0].value.elts] == [item.value for item in second.keywords[0].value.elts]
    first = next(node.value for node in scope.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'model')
    second = next(node.value for node in build.body if isinstance(node, ast.Return))
    assert ast.dump(first) == ast.dump(second)
    for name in ['subModelSVI.reset_classifier', 'subModelSAT.reset_classifier']:
        original_calls = [node for node in ast.walk(author) if isinstance(node, ast.Call)
                          and ast.unparse(node.func) == name]
        current_calls = [node for node in ast.walk(build) if isinstance(node, ast.Call)
                         and ast.unparse(node.func) == name]
        assert [ast.dump(node) for node in original_calls] == [ast.dump(node) for node in current_calls]


def check_reproduction_protocol(config):
    directory = Path(__file__).parent
    source_geo = ast.parse((REPOSITORY / 'Author CVDisaster' / 'geolocalization' / '1_train_cvdisaster.py').read_text(encoding='utf-8'))
    source_est = ast.parse((REPOSITORY / 'Author CVDisaster' / 'disaster_perception_mapping' / '2_cv.py').read_text(encoding='utf-8'))
    combined_geo = ast.parse((directory / 'train_geo.py').read_text(encoding='utf-8'))
    combined_est = ast.parse((directory / 'train_est.py').read_text(encoding='utf-8'))
    est_module = ast.parse((directory / 'est.py').read_text(encoding='utf-8'))

    def calls(tree, name):
        return sorted([node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == name], key=lambda node: node.lineno)

    def dumps(values):
        return sorted(ast.dump(node, include_attributes=False) for node in values)

    for name in ['train', 'evaluate', 'calc_sim']:
        assert dumps(calls(source_geo, name)) == dumps(calls(combined_geo, name)), name
    assert dumps(calls(source_geo, 'get_cosine_schedule_with_warmup')) == dumps(
        calls(combined_geo, 'get_cosine_schedule_with_warmup'))
    loop = next(node for node in ast.walk(combined_geo) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'epoch')
    events = [ast.unparse(node.func) for node in sorted(
        [node for node in ast.walk(loop) if isinstance(node, ast.Call)
         and ast.unparse(node.func) in ['train', 'evaluate', 'calc_sim', 'train_dataloader.dataset.shuffle']],
        key=lambda node: node.lineno)]
    assert events == ['train', 'evaluate', 'calc_sim', 'train_dataloader.dataset.shuffle']
    assert any(isinstance(node, ast.Compare) and ast.unparse(node) == 'r1_test > best_score'
               for node in ast.walk(loop))
    assert len(calls(combined_geo, 'train_dataloader.dataset.shuffle')) == 2
    assert not calls(combined_geo, 'EstCheckpoints')

    for name in ['model.fit', 'model.compile', 'tf.keras.optimizers.SGD', 'EstCheckpoints',
                 'model.load_weights', 'tf.distribute.MirroredStrategy']:
        assert dumps(calls(source_est, name)) == dumps(calls(combined_est, name)), name
    original_schedule = next(node for node in source_est.body
                             if isinstance(node, ast.ClassDef) and node.name == 'WarmUpCosine')
    current_schedule = next(node for node in est_module.body
                            if isinstance(node, ast.ClassDef) and node.name == 'WarmUpCosine')
    assert ast.dump(original_schedule) == ast.dump(current_schedule)
    assert not calls(combined_est, 'tf.keras.utils.set_random_seed')
    assert 'seed' not in config['est']
    assert any(isinstance(node, ast.ImportFrom) and node.module == 'est_checkpoints'
               and any(alias.name == 'EstCheckpoints' for alias in node.names)
               for node in est_module.body)
    assert len(calls(est_module, 'model.predict_on_batch')) == 1
    assert all(next(keyword.value.value for keyword in call.keywords if keyword.arg == 'pretrain')
               for call in calls(est_module, 'GCViTTiny'))
    assert config['checkpoint_policy'] == 'best'
    assert config['geo']['epochs'] == 10 and config['geo']['batch_size'] == 64
    assert config['geo']['lr'] == 0.0005 and config['geo']['weight_decay'] == 0.01
    assert config['est']['epochs'] == 200 and config['est']['batch_size'] == 12
    assert config['est']['lr'] == 0.01 and config['est']['warmup_steps'] == 10
    estimation = ast.parse((directory / 'estimate.py').read_text(encoding='utf-8'))
    assert dumps(calls(estimation, 'tf.distribute.MirroredStrategy')) == dumps(
        calls(source_est, 'tf.distribute.MirroredStrategy'))
    assert len(calls(estimation, 'predict_dataset')) == 2
    assert not calls(estimation, 'model')
    print('REPRODUCTION_TRAIN_EVAL_BEST_RESUME_PROTOCOL_VERIFIED')


def main():
    parser = argparse.ArgumentParser(description='使用真实归档数据核验固定划分、分类统计和Est结构')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--geo-test', required=True)
    parser.add_argument('--catalogs', nargs='+', required=True)
    parser.add_argument('--est-predictions', required=True)
    parser.add_argument('--est-metrics', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    catalog = [row for path in args.catalogs for row in read_csv(path)]
    relatives = [row['svi_relative_path'] for row in catalog]
    assert len(set(relatives)) == len(relatives) == config['expected_total']
    satellites = ['01_Satellite/' + path for path in relatives]
    grounds = ['00_SVI_aligned/' + path for path in relatives]
    train, test = expected_pairs(satellites, grounds, config['split_seed'], config['train_ratio'])
    with Path(args.geo_test).open(encoding='utf-8', newline='') as handle:
        assert test == [tuple(row) for row in csv.reader(handle)]
    samples = read_samples(args.geo_test)
    assert [(sample.satellite, sample.ground) for sample in samples] == test
    train_ids = {PurePosixPath(pair[0]).stem for pair in train}
    test_ids = {str(sample.sample_id) for sample in samples}
    assert len(train) == 2472 and len(test) == 1649 and not train_ids & test_ids
    numpy_ids = np.asarray([sample.sample_id for sample in samples], dtype=np.int64)
    recorded_ids = sample_ids_for_record(numpy_ids)
    assert all(type(sample_id) is int for sample_id in recorded_ids)
    record_path = REPOSITORY / '.local-cache' / 'combined-experiment-1' / f'ids-{uuid.uuid4().hex}.json'
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(record_path, {'sample_ids': recorded_ids})
    assert read_json(record_path)['sample_ids'] == numpy_ids.tolist()
    print('REAL_NUMPY_SAMPLE_IDS_JSON_ROUNDTRIP_VERIFIED', len(recorded_ids))
    labels_by_path = {row['svi_relative_path']: CLASS_NAMES.index(row['label']) for row in catalog}
    for sample in samples:
        assert sample.label == labels_by_path[str(PurePosixPath(*PurePosixPath(sample.raw_ground).parts[1:]))]
    predictions = read_csv(args.est_predictions)
    labels = [int(row['true_label']) for row in predictions]
    values = [int(row['predicted_label']) for row in predictions]
    report = classification_metrics(labels, values)
    archived = read_json(args.est_metrics)
    assert math.isclose(report['accuracy'], archived['accuracy'], abs_tol=1e-12)
    assert math.isclose(report['classification_report']['macro avg']['f1-score'],
                        archived['macro avg']['f1-score'], abs_tol=1e-12)
    same = paired_effect(labels, values, values, config['statistics'])
    assert same['correct_to_incorrect'] == same['incorrect_to_correct'] == 0
    assert same['accuracy_change_percentage_points'] == 0
    np.testing.assert_array_equal(same['change_ci_percentage_points'], [0.0, 0.0])
    check_est_structure()
    check_reproduction_protocol(config)
    for path in Path(__file__).parent.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print('FIXED_SPLIT_VERIFIED train=2472 test=1649 seed=42 ratio=0.6')
    print('ARCHIVED_EST_METRICS_VERIFIED', report['accuracy'])
    print('PAIRED_INPUT_INVARIANCE_AND_EST_STRUCTURE_VERIFIED')


if __name__ == '__main__':
    main()
