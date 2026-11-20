import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import image_path, load_manifest, read_csv, sha256
from conditions import gaussian_image, noise_image, wrong_permutation
from models import policy


def read_split(directory, split):
    load_manifest(directory)
    rows = read_csv(Path(directory) / f'{split}.csv')
    ids = [row['sample_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('目标清单出现重复 ID')
    lookup = {row['sample_id']: row for row in rows}
    adjacent = {sample_id: [] for sample_id in ids}
    for row in read_csv(Path(directory) / f'{split}_neighbors.csv'):
        if row['neighbor_id'] not in lookup or row['sample_id'] not in lookup:
            raise ValueError('邻图清单跨越允许访问的集合')
        if row['neighbor_id'] == row['sample_id']:
            raise ValueError('邻图不能包含当前目标')
        adjacent[row['sample_id']].append(row)
    for values in adjacent.values():
        values.sort(key=lambda item: int(item['rank']))
        if len({row['neighbor_id'] for row in values}) != len(values):
            raise ValueError('同一目标具有重复邻图 ID')
    return rows, lookup, adjacent


@lru_cache(maxsize=16384)
def image_fingerprint(path):
    return sha256(path)


def make_dataset(directory, split, settings, variant, batch_size, condition='paired',
                 condition_seed=42, geo_paths=None, random_neighbors=False):
    rows, lookup, adjacent = read_split(directory, split)
    p = policy(variant)
    satellite_rows = rows
    if condition == 'wrong':
        satellite_rows = [rows[index] for index in wrong_permutation(len(rows), condition_seed)]
    elif condition not in ('paired', 'noise', 'gaussian', 'zero', 'geo'):
        raise ValueError(f'未知卫星输入条件: {condition}')
    if condition == 'geo' and (geo_paths is None or set(geo_paths) != set(lookup)):
        raise ValueError('Geo 卫星清单必须完整覆盖同一测试目标')
    count = settings['neighbors']
    payload = {}
    if p['ground']:
        payload['ground'] = [image_path(settings['data_root'], row['ground']) for row in rows]
    if p['satellite']:
        payload['satellite'] = [image_path(settings['data_root'],
            geo_paths[row['sample_id']] if condition == 'geo' else satellite_rows[index]['satellite'])
            for index, row in enumerate(rows)]
        payload['sample_id'] = [int(row['sample_id']) for row in rows]
    if p['neighbors']:
        paths, masks, relations = [], [], []
        random_order = wrong_permutation(len(rows), condition_seed) if random_neighbors else None
        for row in rows:
            values = adjacent[row['sample_id']]
            if len(values) > count:
                raise ValueError('邻图清单数量超过模型配置')
            selected, valid, relation = [], [], []
            seen_fingerprints = {image_fingerprint(image_path(settings['data_root'], row['ground']))}
            for item in values:
                source = lookup[item['neighbor_id']]
                if random_neighbors:
                    source_index = int(random_order[rows.index(source)])
                    source = rows[source_index]
                    if source['sample_id'] == row['sample_id']:
                        source = rows[(source_index + 1) % len(rows)]
                path = image_path(settings['data_root'], source['ground'])
                fingerprint = image_fingerprint(path)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                selected.append(path)
                valid.append(1.0)
                relation.append([float(item['distance_m']) / settings['radius_m'],
                                 float(item['delta_seconds']) / settings['time_seconds']])
            # 使用当前真实图像填充缺少的位置，显式 mask 使其贡献为零。
            while len(selected) < count:
                selected.append(image_path(settings['data_root'], row['ground']))
                valid.append(0.0)
                relation.append([0.0, 0.0])
            paths.append(selected)
            masks.append(valid)
            relations.append(relation)
        payload.update(neighbors=paths, mask=np.asarray(masks, np.float32),
                       relations=np.asarray(relations, np.float32))
    labels = np.asarray([int(row['label']) for row in rows], np.int64)
    names = ['final', 'base', 'satellite_aux']

    def load(path, shape):
        return tf.image.resize(tf.io.decode_png(tf.io.read_file(path), channels=3), shape[:2])

    def decode(record, label):
        result = {}
        if p['ground']:
            result['ground'] = load(record['ground'], settings['ground_shape'])
        if p['satellite']:
            if condition in ('noise', 'gaussian'):
                generator = noise_image if condition == 'noise' else gaussian_image
                satellite = tf.numpy_function(
                    lambda item: generator(item, condition_seed, settings['satellite_shape']),
                    [record['sample_id']], tf.float32)
                satellite.set_shape(settings['satellite_shape'])
                result['satellite'] = satellite
            elif condition == 'zero':
                result['satellite'] = tf.zeros(settings['satellite_shape'], dtype=tf.float32)
            else:
                result['satellite'] = load(record['satellite'], settings['satellite_shape'])
        if p['neighbors']:
            result['neighbors'] = tf.stack([load(record['neighbors'][index], settings['ground_shape'])
                                            for index in range(count)])
            result['mask'] = record['mask']
            result['relations'] = record['relations']
        return result, {name: label for name in names}

    dataset = tf.data.Dataset.from_tensor_slices((payload, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    dataset = dataset.with_options(options).map(decode, num_parallel_calls=2)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(1), rows
