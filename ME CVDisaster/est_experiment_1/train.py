import argparse
import math
from pathlib import Path
import time

import tensorflow as tf

from common import (ROOT, check_manifest_settings, configuration, environment, load_manifest, new_directory,
                    model_fingerprints, read_json, require_compute_node, sha256, write_json)
from dataset import make_dataset
from models import VARIANTS, build_model, compile_model
from training import AccumulatingModel, Checkpoints, configure_devices


def main():
    parser = argparse.ArgumentParser(description='EST 独立验证选择与完整训练集合重训')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', choices=('select', 'refit'), required=True)
    parser.add_argument('--selection-run', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--gpus', type=int, choices=(0, 2), default=2)
    parser.add_argument('--random-neighbors', action='store_true')
    args = parser.parse_args()
    require_compute_node()
    settings = configuration(args.config)
    if args.seed != settings['seed']:
        raise ValueError('训练种子与实验配置不一致')
    manifest = load_manifest(args.manifest)
    check_manifest_settings(settings, manifest)
    strategy = configure_devices(args.gpus)
    tf.keras.utils.set_random_seed(args.seed)
    stop_epoch = settings['epochs']
    selection_source = None
    if args.stage == 'refit':
        if args.selection_run is None:
            raise ValueError('完整训练集合重训需要明确的验证选择产物')
        selection = read_json(args.selection_run / 'training.json')
        if (selection['stage'] != 'select' or selection['settings'] != settings
                or selection['seed'] != args.seed or selection['variant'] != args.variant
                or selection['manifest_sha256'] != sha256(args.manifest / 'manifest.json')
                or selection['model_source_sha256'] != model_fingerprints()
                or selection['random_neighbors'] != args.random_neighbors):
            raise ValueError('验证选择产物的模型、种子、划分或配置不一致')
        if sha256(args.selection_run / 'selected.h5') != selection['selected_sha256']:
            raise ValueError('验证选择权重指纹不一致')
        stop_epoch = selection['selected_epoch']
        selection_source = {'training_sha256': sha256(args.selection_run / 'training.json'),
                            'selected_epoch': stop_epoch}
    elif args.selection_run is not None:
        raise ValueError('验证选择阶段不能读取其他模型的选择结果')
    split = 'train' if args.stage == 'select' else 'refit'
    training, rows = make_dataset(args.manifest, split, settings, args.variant, settings['batch_size'],
                                 random_neighbors=args.random_neighbors)
    validation = None
    if args.stage == 'select':
        validation, _ = make_dataset(args.manifest, 'validation', settings, args.variant, settings['batch_size'],
                                     random_neighbors=args.random_neighbors)
    steps = math.ceil(len(rows) / settings['batch_size'])
    run_config = {'settings': settings, 'variant': args.variant, 'seed': args.seed,
                  'stage': args.stage, 'stop_epoch': stop_epoch, 'optimizer_steps_per_epoch': steps,
                  'manifest_sha256': sha256(args.manifest / 'manifest.json'),
                  'model_source_sha256': model_fingerprints(),
                  'random_neighbors': args.random_neighbors, 'selection_source': selection_source,
                  'gpus': args.gpus, 'microbatch_per_replica': 1}
    output = new_directory(args.output)
    write_json(output / 'run_config.json', run_config)
    write_json(output / 'environment.json', environment())
    with strategy.scope():
        functional = build_model(settings, args.variant)
        model = AccumulatingModel(functional, len(rows), settings['batch_size'])
        compile_model(model, settings, steps)
        checkpoint = Checkpoints(model, output, run_config, args.stage == 'select', args.resume_from)
    started = time.monotonic()
    if checkpoint.completed < stop_epoch:
        model.fit(training, validation_data=validation, epochs=stop_epoch,
                  initial_epoch=checkpoint.completed,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN(), checkpoint,
                             tf.keras.callbacks.CSVLogger(str(output / 'history.csv'))])
    if checkpoint.completed != stop_epoch or not checkpoint.best_file.is_file():
        raise RuntimeError('训练尚未完成规定轮次或缺少选定权重')
    if int(model.optimizer.iterations.numpy()) != steps * stop_epoch:
        raise ValueError('实际 SGD 更新次数与目标 batch 规则不一致')
    write_json(output / 'training.json', {
        **run_config, 'completed_epochs': checkpoint.completed, 'selected_epoch': checkpoint.best_epoch,
        'validation_accuracy': checkpoint.best if args.stage == 'select' else None,
        'selected_sha256': sha256(checkpoint.best_file), 'parameters': model.count_params(),
        'optimizer_steps': int(model.optimizer.iterations.numpy()),
        'elapsed_seconds': time.monotonic() - started, 'counts': manifest['counts']})


if __name__ == '__main__':
    main()
