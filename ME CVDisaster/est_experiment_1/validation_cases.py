import argparse
from pathlib import Path

from common import read_csv, write_json


def cases_from_manifest(manifest, data_root, weights):
    rows = read_csv(Path(manifest) / 'train.csv')
    lookup = {item['sample_id']: item for item in rows}
    adjacent = read_csv(Path(manifest) / 'train_neighbors.csv')
    neighbors = {key: [] for key in lookup}
    for item in adjacent:
        neighbors[item['sample_id']].append(item['neighbor_id'])
    chosen = []
    for label in ('0', '1', '2'):
        candidates = [item for item in rows if item['label'] == label and len(neighbors[item['sample_id']]) == 2]
        if not candidates:
            raise ValueError('当前训练集合中缺少具有两张邻图的对应类别')
        chosen.append(candidates[0])
    ids = {item['sample_id'] for item in chosen}
    for item in chosen:
        ids.update(neighbors[item['sample_id']])
    return {'data_root': str(data_root), 'weights': str(weights), 'targets': chosen,
            'neighbors': {item['sample_id']: neighbors[item['sample_id']] for item in chosen},
            'samples': {key: lookup[key] for key in ids}}


def main():
    parser = argparse.ArgumentParser(description='从真实训练清单选择模型验证图片')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, cases_from_manifest(args.manifest, args.data_root, args.weights))


if __name__ == '__main__':
    main()
