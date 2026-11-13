import argparse
from collections import Counter
from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split

from common import DEFAULT_CONFIG, new_directory, read_config, sha256, write_csv, write_json
from data import UnifiedSplits


def expected_pairs(satellites, grounds, seed=42, train_ratio=0.6):
    satellites, grounds = sorted(satellites), sorted(grounds)
    if len(satellites) != len(grounds):
        raise ValueError('两种视角的图片数量不同')
    for sat, ground in zip(satellites, grounds):
        if Path(sat).parts[1:] != Path(ground).parts[1:]:
            raise ValueError('两种视角的排序、ID或类别不同')
    return train_test_split(list(zip(satellites, grounds)), train_size=train_ratio, random_state=seed)


def main():
    parser = argparse.ArgumentParser(description='固化Geo与Est共用的seed=42、6:4数据清单')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config['data_root'])
    splits = UnifiedSplits.from_config(config)
    satellites = [path.relative_to(root).as_posix() for path in (root / '01_Satellite').glob('*/*.png')]
    grounds = [path.relative_to(root).as_posix() for path in (root / '00_SVI_aligned').glob('*/*.png')]
    expected = expected_pairs(satellites, grounds, config['split_seed'], config['train_ratio'])
    if len(satellites) != config['expected_total']:
        raise ValueError('图片总数不符合CVIAN配置')
    for samples, pairs in zip([splits.train, splits.test], expected):
        if [(sample.satellite, sample.ground) for sample in samples] != pairs:
            raise ValueError('现有清单与seed=42的6:4划分不同')
        for sample in samples:
            sample.image_paths(root)
    output = new_directory(args.output)
    for name in [config['train_split'], config['test_split']]:
        shutil.copyfile(root / name, output / name)
    for name, samples in [('train', splits.train), ('test', splits.test)]:
        write_csv(output / f'{name}_samples.csv', [vars(sample) for sample in samples])
    write_json(output / 'splits.json', {
        'seed': config['split_seed'], 'train_ratio': config['train_ratio'],
        'data_root': config['data_root'],
        'counts': {'train': len(splits.train), 'test': len(splits.test)},
        'class_counts': {name: dict(Counter(sample.label for sample in samples))
                         for name, samples in [('train', splits.train), ('test', splits.test)]},
        'split_sha256': {name: sha256(output / name)
                         for name in [config['train_split'], config['test_split']]}})
    print('FIXED_SPLITS_READY', len(splits.train), len(splits.test), flush=True)


if __name__ == '__main__':
    main()
