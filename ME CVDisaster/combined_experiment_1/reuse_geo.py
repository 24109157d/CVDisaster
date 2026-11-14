import argparse
from pathlib import Path
import shutil

from common import (DEFAULT_CONFIG, environment, new_directory, read_config, read_json,
                    require_compute_node, sha256, write_json)
from data import load_fixed_splits


def validate_reference(config, reference):
    source = Path(reference['source_run']).resolve(strict=True)
    original_config = source / 'source' / 'ME CVDisaster' / 'combined_experiment_1' / 'config.json'
    if read_json(original_config) != config:
        raise ValueError('Geo原训练配置与当前实验配置不同')
    splits, manifest = load_fixed_splits(config, source / 'splits')
    if manifest['split_sha256'] != reference['split_sha256']:
        raise ValueError('Geo原训练划分与复用记录不同，不能复用权重')
    submission = read_json(reference['submission_record'])
    if (submission['job_id'] != reference['source_job_id']
            or submission['source_commit'] != reference['source_commit']):
        raise ValueError('Geo原训练作业或代码提交记录不同')
    original_script = original_config.with_name('train_geo.py')
    if sha256(original_script) != reference['training_script_sha256']:
        raise ValueError('Geo原训练源码与复用记录不同')
    if sha256(reference['training_log']) != reference['training_log_sha256']:
        raise ValueError('Geo原训练日志发生变化')
    if (reference['completed_epochs'] != config['geo']['epochs']
            or not 0 < reference['best_epoch'] <= reference['completed_epochs']):
        raise ValueError('Geo原训练轮次不符合实验配置')
    checkpoint = (source / 'geo' / reference['checkpoint']).resolve(strict=True)
    final_checkpoint = source / 'geo' / 'weights_end.pth'
    expected_name = f"weights_e{reference['best_epoch']}_{reference['best_score']:.4f}.pth"
    if (checkpoint.parent != source / 'geo' or checkpoint.name != expected_name
            or sha256(checkpoint) != reference['checkpoint_sha256']
            or sha256(final_checkpoint) != reference['final_checkpoint_sha256']):
        raise ValueError('Geo best或训练结束权重与复用记录不同')
    return source, checkpoint, splits, manifest


def main():
    parser = argparse.ArgumentParser(description='复用已完成训练的Geo best及其原始固定划分')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--reference', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--splits-output')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = read_config(args.config)
    reference = read_json(args.reference)
    source, checkpoint, splits, manifest = validate_reference(config, reference)
    print('GEO_REUSE_REFERENCE_VERIFIED', reference['source_job_id'],
          len(splits.train), len(splits.test), reference['checkpoint_sha256'], flush=True)
    if args.check_only:
        return
    require_compute_node()
    if args.splits_output is None or args.output is None:
        raise ValueError('复用权重必须指定新的清单目录和Geo输出目录')
    splits_output = Path(args.splits_output).resolve()
    shutil.copytree(source / 'splits', splits_output)
    _, copied_manifest = load_fixed_splits(config, splits_output)
    if copied_manifest != manifest:
        raise ValueError('复制后的固定划分与Geo原训练划分不同')
    output = new_directory(args.output)
    copied_checkpoint = output / checkpoint.name
    shutil.copyfile(checkpoint, copied_checkpoint)
    if sha256(copied_checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('复制后的Geo best权重SHA256不同')
    write_json(output / 'training.json', {
        'stage': 'geo', 'config': config, 'split_sha256': manifest['split_sha256'],
        'checkpoint_policy': 'best', 'best_epoch': reference['best_epoch'],
        'best_score': reference['best_score'], 'completed_epochs': reference['completed_epochs'],
        'checkpoint': copied_checkpoint.name, 'checkpoint_sha256': reference['checkpoint_sha256'],
        'initial_sha256': config['geo']['initial_sha256'],
        'record_type': 'reused_geo', 'training_source': reference, 'reuse_environment': environment()})
    print('GEO_BEST_AND_ORIGINAL_SPLITS_REUSED', output, splits_output, flush=True)


if __name__ == '__main__':
    main()
