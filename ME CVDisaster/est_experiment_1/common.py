import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import uuid


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ('0_MinorDamage', '1_ModerateDamage', '2_SevereDamage')
SPLIT_KEYS = ('split_protocol', 'split_seed', 'train_ratio', 'validation_seed', 'validation_fraction')
EST_REFERENCE_SHA256 = {
    'train': 'd4c8793214d2f99878232b54dfefa9c040c8d48e2c1f657ff38a4b457d5abb5a',
    'test': '516a07589b7f41a889a209a7b3852e4b81c0ae6cc290a410273f6ff1c504c41c'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, replace=False):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if not replace:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(payload)
        return
    temporary = path.with_name(f'.{path.name}-{uuid.uuid4().hex}')
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError('CSV 记录不能为空')
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def configuration(path=ROOT / 'config.json'):
    value = read_json(path)
    if (value['split_protocol'] != 'est626' or value['split_seed'] != 626
            or value['train_ratio'] != 0.6 or value['validation_seed'] != 626
            or value['validation_fraction'] != 0.2):
        raise ValueError('当前实验要求 EST seed=626 的 6:4 划分及 seed=626 的内部验证划分')
    if value['epochs'] != 200 or value['batch_size'] != 12:
        raise ValueError('正式比较要求 200 epochs 和有效 batch size 12')
    if (value['dimension'] <= 0 or value['heads'] <= 0 or value['attention_dimension'] <= 0
            or value['attention_dimension'] % value['heads'] or value['neighbors'] < 1):
        raise ValueError('特征维度、attention heads 或邻图数量无效')
    if not 0 <= value['redundancy_threshold'] < 1:
        raise ValueError('重复内容阈值必须位于 [0,1)')
    if min(value['lambda_base'], value['lambda_satellite'], value['lambda_ordinal']) < 0:
        raise ValueError('辅助监督权重不能小于零')
    if value['ground_shape'] != [512, 1024, 3] or value['satellite_shape'] != [512, 512, 3]:
        raise ValueError('当前实验要求街景 512×1024 和卫星 512×512')
    if not 0 < value['local_crop_size'] < 512 or value['local_crop_size'] % 32:
        raise ValueError('卫星中心裁切尺寸必须小于 512，并为 32 的整数倍')
    if not 0 < value['film_strength'] <= 1:
        raise ValueError('FiLM 残差强度必须位于 (0,1]')
    if not 0 <= value['gate_start_epoch'] < value['gate_full_epoch'] < value['epochs']:
        raise ValueError('门控预热轮次无效')
    if len(value['ordinal_boundary_weights']) != 2 or min(value['ordinal_boundary_weights']) <= 0:
        raise ValueError('有序损失需要两个正数分界权重')
    if value['radius_m'] <= 0 or value['time_seconds'] <= 0:
        raise ValueError('邻图距离和时间范围必须大于零')
    return value


def image_path(root, relative):
    relative = PurePosixPath(relative)
    root = Path(root).resolve()
    if (len(relative.parts) != 3 or relative.parts[0] not in ('00_SVI', '01_Satellite')
            or relative.parts[1] not in CLASS_NAMES or relative.suffix != '.png'
            or not relative.stem.isdecimal()):
        raise ValueError(f'图片路径无效: {relative}')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.stat().st_size == 0:
        raise ValueError(f'图片文件无效: {path}')
    return str(path)


def environment():
    commit = os.environ['CVDISASTER_COMMIT']
    if re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('作业源码需要明确的完整 Git 提交号')
    return {
        'python': sys.version, 'platform': platform.platform(), 'argv': sys.argv,
        'commit': commit,
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': source_fingerprints(),
        'packages': {item.metadata['Name']: item.version for item in importlib.metadata.distributions()},
    }


def source_fingerprints():
    return {path.name: sha256(path) for path in sorted(ROOT.glob('*.py'))}


def model_fingerprints():
    return {name: sha256(ROOT / name) for name in ('models.py', 'dataset.py', 'conditions.py', 'training.py', 'common.py')}


def require_compute_node():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('正式训练与评价需要 Slurm 计算节点')


def load_manifest(directory):
    directory = Path(directory).resolve()
    metadata = read_json(directory / 'manifest.json')
    for name, digest in metadata['files'].items():
        path = directory / name
        if path.parent != directory or sha256(path) != digest:
            raise ValueError(f'固定清单指纹不一致: {name}')
    return metadata


def check_manifest_settings(settings, manifest):
    if (manifest['protocol'] != settings['split_protocol']
            or manifest['split_settings'] != {key: settings[key] for key in SPLIT_KEYS}
            or manifest['reference_splits_sha256'] != EST_REFERENCE_SHA256):
        raise ValueError('模型配置与复现 EST 的固定划分不一致')
    expected = {key: settings[key] for key in ('neighbors', 'radius_m', 'time_seconds', 'past_only')}
    if manifest['neighbor_settings'] != expected:
        raise ValueError('模型配置与固定邻图规则不一致')
