import argparse
import gc
from pathlib import Path, PurePosixPath

from common import (DEFAULT_CONFIG, environment, new_directory, read_config, require_compute_node,
                    sha256, write_csv, write_json)
from data import load_fixed_splits
from est import (EstCheckpoints, WarmUpCosine, build_model, check_pretrained_cache,
                 configure_gpus, make_dataset, predict_dataset)

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser(description='合并实验1：按复现流程独立训练Est')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--splits', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume-from', type=Path,
                        help='旧Est输出目录，必须包含完整latest/current.json')
    args = parser.parse_args()
    require_compute_node()
    settings = read_config(args.config)
    params = settings['est']
    splits, manifest = load_fixed_splits(settings, args.splits)
    configure_gpus(params['gpus'])
    check_pretrained_cache(params)
    output = new_directory(args.output)
    split_manifest = {}
    for split_name, samples in [('train', splits.train), ('test', splits.test)]:
        split_file = output / f'est_{split_name}_60.csv'
        write_csv(split_file, [{'svi_relative_path': str(PurePosixPath(*PurePosixPath(sample.raw_ground).parts[1:])),
                                'label': PurePosixPath(sample.raw_ground).parent.name} for sample in samples])
        split_manifest[split_name] = {'count': len(samples), 'file': split_file.name,
                                      'sha256': sha256(split_file)}
    trData = make_dataset(splits.train, settings['data_root'], params['batch_size'])
    vaData = make_dataset(splits.test, settings['data_root'], params['batch_size'])
    STEPS_PER_EPOCH = int(trData.cardinality().numpy())
    EPOCHS = params['epochs']
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    run_config = {
        'protocol': 'github-est-sgd', 'backbone': params['backbone'],
        'train_ratio': settings['train_ratio'], 'split_seed': settings['split_seed'], 'gpus': params['gpus'],
        'batch_size': params['batch_size'], 'epochs': EPOCHS, 'optimizer': 'SGD',
        'base_lr': params['lr'], 'warmup_lr': params['warmup_lr'], 'weight_decay': None,
        'momentum': 0.0, 'nesterov': False, 'steps_per_epoch': STEPS_PER_EPOCH,
        'warmup_steps': params['warmup_steps'], 'total_steps': TOTAL_STEPS,
        'weight_decay_exclusions': [], 'splits': split_manifest,
    }
    write_json(output / 'est_config.json', run_config)
    strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0', '/GPU:1'])
    with strategy.scope():
        lr_schedule = WarmUpCosine(learning_rate_base=params['lr'], total_steps=TOTAL_STEPS,
                                   warmup_learning_rate=params['warmup_lr'], warmup_steps=params['warmup_steps'])
        optimizer = tf.keras.optimizers.SGD(lr_schedule)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model = build_model()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        for variable in model.trainable_variables:
            replicas = strategy.experimental_local_results(variable)
            if len(replicas) != 2 or any(
                    f'GPU:{index}' not in replica.device
                    for index, replica in enumerate(replicas)):
                raise RuntimeError(f'Est 权重未正确镜像到两张 GPU: {variable.name}')
    ckpt_path = str(output / '0.6_bestCV.h5')
    ckpt_callback = EstCheckpoints(model, run_config, ckpt_path, args.resume_from)
    if ckpt_callback.initial_epoch < EPOCHS:
        history = model.fit(trData,
            validation_data=vaData,
            epochs=EPOCHS,
            initial_epoch=ckpt_callback.initial_epoch,
            callbacks=[ckpt_callback])
        write_json(output / 'history.json', history.history)
    else:
        print('EST_TRAINING_ALREADY_COMPLETE', EPOCHS, flush=True)
    model.load_weights(ckpt_path)
    del trData
    gc.collect()
    yTrue, probabilities = predict_dataset(model, vaData)
    yPred = np.argmax(probabilities, axis=1)
    if len(yTrue) != len(splits.test):
        raise ValueError('Est最终测试必须覆盖全部固定测试样本')
    print(classification_report(yTrue, yPred, target_names=['minor', 'moderate', 'severe'], digits=5))
    overallP = precision_score(yTrue, yPred, average='macro')
    overallR = recall_score(yTrue, yPred, average='macro')
    print('Overall Precision:', overallP, '\nOverall Recall:', overallR)
    report = classification_report(yTrue, yPred, target_names=['minor', 'moderate', 'severe'], output_dict=True)
    report.update(best_epoch=int(ckpt_callback.best_epoch.numpy()),
                  best_val_accuracy=float(ckpt_callback.best_score.numpy()),
                  completed_epochs=ckpt_callback.initial_epoch,
                  optimizer_steps=int(optimizer.iterations.numpy()), checkpoint=Path(ckpt_path).name)
    write_json(output / 'est_metrics.json', report)
    write_csv(output / 'est_predictions.csv', [
        {'svi_relative_path': str(PurePosixPath(*PurePosixPath(sample.raw_ground).parts[1:])),
         'true_label': int(yTrue[index]), 'predicted_label': int(yPred[index])}
        for index, sample in enumerate(splits.test)])
    load_fixed_splits(settings, args.splits)
    write_json(output / 'training.json', {
        'stage': 'est', 'config': settings, 'split_sha256': manifest['split_sha256'],
        'checkpoint_policy': 'best', 'best_epoch': report['best_epoch'],
        'best_score': report['best_val_accuracy'], 'checkpoint': Path(ckpt_path).name,
        'checkpoint_sha256': sha256(ckpt_path), 'initial_sha256': sha256(params['initial_weights']),
        'completed_epochs': ckpt_callback.initial_epoch, 'steps_per_epoch': STEPS_PER_EPOCH,
        'optimizer_steps': int(optimizer.iterations.numpy()), 'environment': environment()})


if __name__ == '__main__':
    main()
