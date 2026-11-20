import argparse
import csv
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit

from common import (CLASS_NAMES, EST_REFERENCE_SHA256, ROOT, SPLIT_KEYS, configuration,
                    new_directory, read_csv, read_json, sha256, write_csv, write_json)


def read_metadata(path):
    features = read_json(path)['features']
    records = []
    for feature in features:
        p = feature['properties']
        records.append({'sample_id': str(p['id']), 'sequence': str(p['sequence_i']),
                        'timestamp': float(p['captured_a']) / 1000.0,
                        'longitude': float(p['lon']), 'latitude': float(p['lat'])})
    frame = pd.DataFrame(records).set_index('sample_id', verify_integrity=True)
    if len(frame) != 4121 or frame['sequence'].isin(['', 'None']).any():
        raise ValueError('元数据必须包含 4,121 个唯一 ID 和有效 Sequence')
    if not np.isfinite(frame[['timestamp', 'longitude', 'latitude']].to_numpy()).all():
        raise ValueError('元数据包含非有限坐标或时间')
    return frame


def read_est_split(path, metadata):
    records = []
    for item in read_csv(path):
        if set(item) != {'svi_relative_path', 'label'}:
            raise ValueError('EST 清单必须包含 svi_relative_path、label 两个字段')
        relative = PurePosixPath(item['svi_relative_path'])
        if (len(relative.parts) != 2 or relative.parts[0] not in CLASS_NAMES
                or relative.suffix != '.png' or not relative.stem.isdecimal()
                or item['label'] != relative.parts[0]):
            raise ValueError('复现 EST 清单中的图片路径或类别无效')
        sample_id = relative.stem
        records.append({'sample_id': sample_id, 'label': CLASS_NAMES.index(item['label']),
                        'ground': f'00_SVI/{relative.as_posix()}',
                        'satellite': f'01_Satellite/{relative.as_posix()}',
                        **metadata.loc[sample_id].to_dict()})
    frame = pd.DataFrame(records)
    if frame['sample_id'].duplicated().any():
        raise ValueError('数据划分存在重复 ID')
    return frame


def validation_split(frame, seed, fraction=0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train, validation = next(splitter.split(frame, groups=frame['sequence']))
    for indices in (train, validation):
        if set(frame.iloc[indices]['label']) != {0, 1, 2}:
            raise ValueError('训练或验证集合缺少类别，需重新审查固定划分')
    return frame.iloc[train].copy(), frame.iloc[validation].copy()


def neighbors(frame, settings):
    geod = Geod(ellps='WGS84')
    rows = []
    for _, target in frame.iterrows():
        candidates = frame[(frame['sequence'] == target['sequence'])
                           & (frame['sample_id'] != target['sample_id'])].copy()
        dt = candidates['timestamp'].to_numpy() - target['timestamp']
        _, _, distance = geod.inv(np.full(len(candidates), target['longitude']),
                                  np.full(len(candidates), target['latitude']),
                                  candidates['longitude'].to_numpy(), candidates['latitude'].to_numpy())
        candidates['distance'] = distance
        candidates['delta_time'] = dt
        candidates['abs_time'] = np.abs(dt)
        valid = (distance <= settings['radius_m']) & (np.abs(dt) <= settings['time_seconds'])
        if settings['past_only']:
            valid &= dt <= 0
        candidates = candidates[valid].sort_values(['distance', 'abs_time', 'sample_id'])
        for rank, (_, neighbor) in enumerate(candidates.head(settings['neighbors']).iterrows()):
            rows.append({'sample_id': target['sample_id'], 'neighbor_id': neighbor['sample_id'],
                         'rank': rank, 'distance_m': float(neighbor['distance']),
                         'delta_seconds': float(neighbor['delta_time'])})
    return rows


def spatial_groups(frame):
    project = Transformer.from_crs(4326, 32617, always_xy=True)
    east, north = project.transform(frame['longitude'].to_numpy(), frame['latitude'].to_numpy())
    grid = np.floor(np.column_stack([east, north]) / 1000).astype(np.int64)
    _, cells = np.unique(grid, axis=0, return_inverse=True)
    edges = []
    for _, indices in frame.groupby('sequence', sort=False).indices.items():
        unique = np.unique(cells[indices])
        edges.extend((int(unique[0]), int(cell)) for cell in unique)
    row, col = np.array(edges).T
    count = int(cells.max()) + 1
    graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    return labels[cells]


def publish(directory, sets, settings, sources):
    output = new_directory(directory)
    files = {}
    counts = {}
    for name, frame in sets.items():
        if frame.empty or set(frame['label']) != {0, 1, 2}:
            raise ValueError(f'{name} 为空或缺少类别')
        name_csv = f'{name}.csv'
        write_csv(output / name_csv, frame.to_dict('records'))
        files[name_csv] = sha256(output / name_csv)
        adjacent = neighbors(frame, settings)
        neighbor_file = output / f'{name}_neighbors.csv'
        with neighbor_file.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'sample_id', 'neighbor_id', 'rank', 'distance_m', 'delta_seconds'])
            writer.writeheader()
            writer.writerows(adjacent)
        files[neighbor_file.name] = sha256(neighbor_file)
        covered = {item['sample_id'] for item in adjacent}
        counts[name] = {'targets': len(frame), 'with_neighbors': len(covered),
                        'sequences': int(frame['sequence'].nunique())}
    write_json(output / 'manifest.json', {'protocol': settings['split_protocol'], 'files': files, 'counts': counts,
                                        'split_settings': {key: settings[key] for key in SPLIT_KEYS},
                                        'reference_splits_sha256': EST_REFERENCE_SHA256,
                                        'neighbor_settings': {key: settings[key] for key in (
                                            'neighbors', 'radius_m', 'time_seconds', 'past_only')},
                                        'sources': sources, 'spatial_independence': False})
    return str(output)


def main():
    parser = argparse.ArgumentParser(description='沿用复现 EST seed=626 的实际 6:4 清单')
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--est-train', type=Path, required=True)
    parser.add_argument('--est-test', type=Path, required=True)
    parser.add_argument('--est-config', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = configuration(args.config)
    original = read_json(args.est_config)
    if (original['protocol'] != 'github-est-sgd' or original['split_seed'] != settings['split_seed']
            or original['train_ratio'] != settings['train_ratio']):
        raise ValueError('参考配置与复现 EST seed=626 的 6:4 划分不一致')
    for name, path, count in [('train', args.est_train, 2472), ('test', args.est_test, 1649)]:
        expected = original['splits'][name]
        if (sha256(path) != EST_REFERENCE_SHA256[name]
                or expected['sha256'] != EST_REFERENCE_SHA256[name] or expected['count'] != count):
            raise ValueError(f'复现 EST 的 {name} 清单指纹或数量不一致')
    metadata = read_metadata(args.metadata)
    train = read_est_split(args.est_train, metadata)
    test = read_est_split(args.est_test, metadata)
    full = pd.concat([train, test], ignore_index=True)
    if (len(train) != 2472 or len(test) != 1649 or full['sample_id'].nunique() != 4121
            or set(full['sample_id']) != set(metadata.index)):
        raise ValueError('复现 EST 划分必须互斥并完整覆盖 4,121 个样本')
    sources = {str(path): sha256(path) for path in (
        args.metadata, args.est_train, args.est_test, args.est_config)}
    inner_train, validation = validation_split(train, settings['validation_seed'], settings['validation_fraction'])
    publish(args.output, {'train': inner_train, 'validation': validation,
                          'refit': train, 'test': test}, settings, sources)


if __name__ == '__main__':
    main()
