import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
CLASS_NAMES = ['0_MinorDamage', '1_ModerateDamage', '2_SevereDamage']
DEFAULT_CONFIG = Path(__file__).with_name('config.json')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(payload)


def sample_ids_for_record(samples):
    return [int(sample_id) for sample_id in samples]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError('输出样本不能为空')
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_path(data_root, relative, directory):
    relative = PurePosixPath(relative)
    root = Path(data_root).resolve()
    path = (root / relative).resolve()
    if (len(relative.parts) != 3 or relative.parts[0] != directory
            or relative.parts[1] not in CLASS_NAMES or relative.suffix != '.png'
            or not relative.stem.isdecimal() or not path.is_relative_to(root)):
        raise ValueError(f'图片路径不符合要求: {relative}')
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def read_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    if (config['split_seed'] != 42 or config['train_ratio'] != 0.6
            or config['checkpoint_policy'] != 'best'
            or config['train_split'] != 'train_60.csv' or config['test_split'] != 'val_60.csv'):
        raise ValueError('本实验使用seed=42、6:4划分和复现代码的best权重选择')
    return config


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_compute_node():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('模型训练与推理必须通过Slurm在计算节点执行')


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'argv': sys.argv,
            'commit': os.environ['CVDISASTER_COMMIT'],
            'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ['SLURMD_NODENAME'],
            'packages': {item.metadata['Name']: item.version
                         for item in importlib.metadata.distributions()}}


def verify_training(run, stage, config, manifest):
    run = Path(run)
    metadata = read_json(run / 'training.json')
    if metadata['stage'] != stage or metadata['config'] != config:
        raise ValueError(f'{stage}训练配置不一致')
    if metadata['split_sha256'] != manifest['split_sha256']:
        raise ValueError(f'{stage}训练使用了不同的数据划分')
    if metadata['completed_epochs'] != config[stage]['epochs']:
        raise ValueError(f'{stage}训练尚未完成规定轮次')
    if metadata['checkpoint_policy'] != 'best' or not 0 < metadata['best_epoch'] <= metadata['completed_epochs']:
        raise ValueError(f'{stage}没有有效的best checkpoint记录')
    checkpoint = run / metadata['checkpoint']
    if checkpoint.parent.resolve() != run.resolve() or sha256(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError(f'{stage}权重路径或SHA256不一致')
    return checkpoint, metadata
