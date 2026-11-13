import sys
from types import SimpleNamespace

from common import REPOSITORY

sys.path.insert(0, str(REPOSITORY / 'Author CVDisaster' / 'geolocalization'))

import torch
from torch.utils.data import DataLoader
from sample4geo.dataset.cvdisaster import CVDisasterDatasetEval, CVDisasterDatasetTrain
from sample4geo.model import TimmModel
from sample4geo.transforms import get_transforms_train, get_transforms_val


def require_gpus(count):
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise RuntimeError(f'Geo需要{count}张可见CUDA GPU')


def create_model(config):
    return TimmModel(config['geo']['backbone'], pretrained=True, img_size=384)


def make_loaders(config, model, training=False):
    params = config['geo']
    information = model.get_config()
    sizes = ((384, 384), (round(224 / 1232 * 768), 768))
    datasets, loaders = {}, {}
    if training:
        sat, ground = get_transforms_train(*sizes, mean=information['mean'], std=information['std'])
        datasets['training'] = CVDisasterDatasetTrain(
            config['data_root'], transforms_query=ground, transforms_reference=sat,
            prob_flip=params['prob_flip'], prob_rotate=params['prob_rotate'],
            shuffle_batch_size=params['batch_size'], split_key='val60')
        loaders['training'] = DataLoader(datasets['training'], batch_size=params['batch_size'],
                                       num_workers=params['num_workers'], shuffle=False, pin_memory=True)
    sat_val, ground_val = get_transforms_val(*sizes, mean=information['mean'], std=information['std'])
    evaluation_order = [('test', 'reference'), ('test', 'query')]
    if training:
        evaluation_order.extend([('train', 'query'), ('train', 'reference')])
    for split, view in evaluation_order:
        key = split + '_' + view
        transform = sat_val if view == 'reference' else ground_val
        datasets[key] = CVDisasterDatasetEval(
            config['data_root'], split, view, transform, split_key='val60')
        loaders[key] = DataLoader(
            datasets[key], batch_size=params['eval_batch_size'], num_workers=params['num_workers'],
            shuffle=False, pin_memory=True)
    return datasets, loaders


def runtime_config(config):
    return SimpleNamespace(device='cuda', verbose=True, normalize_features=True,
                           gpu_ids=tuple(range(config['geo']['gpus'])), mixed_precision=True,
                           clip_grad=config['geo']['clip_grad'], scheduler='cosine',
                           neighbour_range=config['geo']['neighbour_range'])
