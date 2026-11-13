import argparse
import time

from common import (DEFAULT_CONFIG, environment, new_directory, read_config, require_compute_node,
                    sha256, write_json)
from data import load_fixed_splits
from geo import create_model, make_loaders, require_gpus, runtime_config

import torch
from torch.cuda.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup
from sample4geo.evaluate.cvusa_and_cvact import calc_sim, evaluate
from sample4geo.loss import InfoNCE
from sample4geo.trainer import train
from sample4geo.utils import setup_system


def main():
    parser = argparse.ArgumentParser(description='合并实验1：按复现流程独立训练Geo')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--splits', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_compute_node()
    settings = read_config(args.config)
    params = settings['geo']
    splits, manifest = load_fixed_splits(settings, args.splits)
    require_gpus(params['gpus'])
    if sha256(params['initial_checkpoint']) != params['initial_sha256']:
        raise ValueError('CVUSA初始权重SHA256不一致')
    output = new_directory(args.output)
    setup_system(seed=params['seed'], cudnn_benchmark=True, cudnn_deterministic=False)
    model = create_model(settings)
    model.load_state_dict(torch.load(params['initial_checkpoint']), strict=False)
    model = torch.nn.DataParallel(model, device_ids=list(range(params['gpus'])))
    model = model.to('cuda')
    datasets, loaders = make_loaders(settings, model.module, training=True)
    for split, samples in [('train', splits.train), ('test', splits.test)]:
        for view in ['query', 'reference']:
            if list(datasets[split + '_' + view].label) != [sample.sample_id for sample in samples]:
                raise ValueError(f'Geo Dataset顺序与固定清单不同: {split}/{view}')
    config = runtime_config(settings)
    train_dataloader = loaders['training']
    reference_dataloader_test = loaders['test_reference']
    query_dataloader_test = loaders['test_query']
    reference_dataloader_train = loaders['train_reference']
    query_dataloader_train = loaders['train_query']
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])
    loss_function = InfoNCE(loss_function=loss_fn, device=config.device)
    scaler = GradScaler(init_scale=2. ** 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    train_steps = len(train_dataloader) * params['epochs']
    warmup_steps = len(train_dataloader) * params['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)

    initial_r1 = evaluate(config=config, model=model, reference_dataloader=reference_dataloader_test,
                          query_dataloader=query_dataloader_test, ranks=[1, 5, 10],
                          step_size=1000, cleanup=True)
    r1_train, sim_dict = calc_sim(config=config, model=model,
                                  reference_dataloader=reference_dataloader_train,
                                  query_dataloader=query_dataloader_train,
                                  ranks=[1, 5, 10], step_size=1000, cleanup=True)
    train_dataloader.dataset.shuffle(sim_dict, neighbour_select=params['neighbour_select'],
                                     neighbour_range=params['neighbour_range'])
    best_score = 0
    best_checkpoint = None
    best_epoch = None
    history = []
    for epoch in range(1, params['epochs'] + 1):
        started = time.monotonic()
        sample_ids = list(train_dataloader.dataset.samples)
        train_loss = train(config, model, dataloader=train_dataloader, loss_function=loss_function,
                           optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        r1_test = evaluate(config=config, model=model, reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, ranks=[1, 5, 10],
                           step_size=1000, cleanup=True)
        r1_train, sim_dict = calc_sim(config=config, model=model,
                                      reference_dataloader=reference_dataloader_train,
                                      query_dataloader=query_dataloader_train,
                                      ranks=[1, 5, 10], step_size=1000, cleanup=True)
        if r1_test > best_score:
            best_score = r1_test
            best_epoch = epoch
            best_checkpoint = output / f'weights_e{epoch}_{r1_test:.4f}.pth'
            torch.save(model.module.state_dict(), best_checkpoint)
        train_dataloader.dataset.shuffle(sim_dict, neighbour_select=params['neighbour_select'],
                                         neighbour_range=params['neighbour_range'])
        history.append({'epoch': epoch, 'loss': float(train_loss), 'sample_ids': sample_ids,
                        'test_r1': float(r1_test), 'train_r1': float(r1_train),
                        'seconds': time.monotonic() - started, 'scheduler_steps': scheduler.last_epoch})
        print('GEO_EPOCH', epoch, 'LOSS', train_loss, 'TEST_R1', r1_test, flush=True)
    torch.save(model.module.state_dict(), output / 'weights_end.pth')
    if best_checkpoint is None:
        raise RuntimeError('训练未产生R@1大于0的最佳checkpoint')
    best_state = torch.load(best_checkpoint, map_location='cpu')
    model.module.load_state_dict(best_state)
    final_r1 = evaluate(config=config, model=model, reference_dataloader=reference_dataloader_test,
                        query_dataloader=query_dataloader_test, ranks=[1, 5, 10],
                        step_size=1000, cleanup=True)
    load_fixed_splits(settings, args.splits)
    write_json(output / 'history.json', history)
    write_json(output / 'training.json', {
        'stage': 'geo', 'config': settings, 'split_sha256': manifest['split_sha256'],
        'checkpoint_policy': 'best', 'best_epoch': best_epoch, 'best_score': float(best_score),
        'checkpoint': best_checkpoint.name, 'checkpoint_sha256': sha256(best_checkpoint),
        'initial_sha256': params['initial_sha256'], 'completed_epochs': params['epochs'],
        'initial_r1': float(initial_r1), 'best_reloaded_r1': float(final_r1),
        'scheduled_total_steps': train_steps, 'scheduler_steps': scheduler.last_epoch,
        'environment': environment()})


if __name__ == '__main__':
    main()
