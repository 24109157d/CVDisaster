import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from common import CLASS_NAMES, image_path, read_json, sha256


@dataclass(frozen=True)
class Sample:
    sample_id: int
    label: int
    satellite: str
    ground: str
    raw_ground: str

    def image_paths(self, data_root):
        return {
            'satellite': image_path(data_root, self.satellite, '01_Satellite'),
            'ground': image_path(data_root, self.ground, '00_SVI_aligned'),
            'raw_ground': image_path(data_root, self.raw_ground, '00_SVI'),
        }


def read_samples(path):
    samples = []
    seen = set()
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 2:
                raise ValueError(f'{path}:{number} 必须包含卫星与街景两列路径')
            satellite, ground = (PurePosixPath(value) for value in row)
            for relative, directory in [(satellite, '01_Satellite'), (ground, '00_SVI_aligned')]:
                if (len(relative.parts) != 3 or relative.parts[0] != directory
                        or relative.parts[1] not in CLASS_NAMES or relative.suffix != '.png'
                        or not relative.stem.isdecimal()):
                    raise ValueError(f'{path}:{number} 图片路径不符合要求: {relative}')
            if satellite.parts[1:] != ground.parts[1:]:
                raise ValueError(f'{path}:{number} 图片ID或类别不一致')
            sample_id = int(satellite.stem)
            if sample_id in seen:
                raise ValueError(f'{path}:{number} 图片ID重复: {sample_id}')
            seen.add(sample_id)
            samples.append(Sample(
                sample_id=sample_id, label=CLASS_NAMES.index(satellite.parent.name),
                satellite=str(satellite), ground=str(ground),
                raw_ground=str(PurePosixPath('00_SVI', *ground.parts[1:]))))
    if not samples:
        raise ValueError(f'数据清单为空: {path}')
    return tuple(samples)


@dataclass(frozen=True)
class UnifiedSplits:
    train: tuple[Sample, ...]
    test: tuple[Sample, ...]

    @classmethod
    def from_csv(cls, train_csv, test_csv):
        train = read_samples(train_csv)
        test = read_samples(test_csv)
        overlap = {sample.sample_id for sample in train} & {sample.sample_id for sample in test}
        if overlap:
            raise ValueError(f'训练集与测试集包含相同图片ID，共{len(overlap)}个')
        return cls(train=train, test=test)

    @classmethod
    def from_config(cls, config):
        root = Path(config['data_root'])
        return cls.from_csv(root / config['train_split'], root / config['test_split'])


def load_fixed_splits(config, directory):
    directory = Path(directory)
    manifest = read_json(directory / 'splits.json')
    if (manifest['seed'] != config['split_seed'] or manifest['train_ratio'] != config['train_ratio']
            or manifest['data_root'] != config['data_root']):
        raise ValueError('固定划分的参数或数据目录与配置不一致')
    for name, expected in manifest['split_sha256'].items():
        if sha256(directory / name) != expected or sha256(Path(config['data_root']) / name) != expected:
            raise ValueError(f'固定划分或共享数据清单发生变化: {name}')
    splits = UnifiedSplits.from_csv(directory / config['train_split'], directory / config['test_split'])
    expected_train = int(config['expected_total'] * config['train_ratio'])
    if len(splits.train) != expected_train or len(splits.test) != config['expected_total'] - expected_train:
        raise ValueError('训练与测试样本数量不符合固定6:4划分')
    return splits, manifest
