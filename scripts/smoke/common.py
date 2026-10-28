"""仅供 smoke test 使用：限制写入范围，并保留每个阶段的成功或失败证据。"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path('/scratch/liuzeyu/heyinxi')
CLASSES = ('0_MinorDamage', '1_ModerateDamage', '2_SevereDamage')
ASSETS = {
    'cvusa': (RUNTIME / 'pretrained/weights_e40_98.6830.pth',
              '739cf00d303836fba1195f1bb8d54c4b13c53926ce3212fc9ceced335fc59a23'),
    'convnext': (RUNTIME / 'pretrained/huggingface/hub/models--timm--convnext_base.fb_in22k_ft_in1k_384/snapshots/2b2698d86523262ca8065d0957c7f80d963bb43f/model.safetensors',
                 'bc0cfafc7d7755e8271f51343c4374725bf6409c0434e64028a863c4aca7d9fe'),
    'gcvit': (RUNTIME / 'pretrained/keras/datasets/gcvit_tiny_weights.h5',
              '90bf9bfaafd086a831e92edbbfbba9450fd56786c2d39b3ac3f215d0cef54768'),
}


def digest(path):
    """以大块顺序读取校验文件，不修改源文件。"""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Report:
    def __init__(self, module, output):
        self.output = output
        self.state = {'module': module, 'status': 'running', 'stages': [],
                      'code_commit': os.environ.get('CVDISASTER_COMMIT'),
                      'limitations': '仅验证少量样本和一个训练 batch，不代表完整训练或正式指标。'}
        self.save()

    def save(self):
        (self.output / 'summary.json').write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    @contextmanager
    def stage(self, name):
        # 在耗时导入或 GPU 操作之前落盘，超时后仍可定位停在哪一步。
        record = {'name': name, 'status': 'running'}
        self.state['stages'].append(record)
        self.save()
        print('STAGE_START', name, flush=True)
        start = time.monotonic()
        try:
            yield
        except BaseException:
            record['status'] = 'failed'
            raise
        else:
            record['status'] = 'passed'
        finally:
            record['seconds'] = round(time.monotonic() - start, 3)
            self.save()
            print('STAGE_END', name, record['status'], flush=True)


def pairs(root, count):
    """按类别轮流选少量样本；同相对路径配对，不按两个文件列表的位置猜配对。"""
    assert sorted(p.name for p in (root / '00_SVI').iterdir() if p.is_dir()) == list(CLASSES)
    assert sorted(p.name for p in (root / '01_Satellite').iterdir() if p.is_dir()) == list(CLASSES)
    files = [sorted((root / '00_SVI' / name).glob('*.png')) for name in CLASSES]
    selected = []
    for index in range(count):
        label = index % len(CLASSES)
        svi = files[label][index // len(CLASSES)]
        relative = svi.relative_to(root / '00_SVI')
        sat = root / '01_Satellite' / relative
        assert sat.is_file(), f'缺少同名卫星图像：{sat}'
        assert svi.stem.isdecimal(), f'图像 ID 不是整数：{svi.name}'
        selected.append({'id': int(svi.stem), 'label': label,
                         'svi': str(svi), 'sat': str(sat), 'relative': str(relative)})
    assert len({p['id'] for p in selected}) == count, '所选样本 ID 重复'
    return selected


def execute(module, work):
    parser = argparse.ArgumentParser(description=f'{module} 单 batch GPU smoke test')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # 独立调用 Python 时也要求正确离线缓存配置，避免模型工厂触发联网下载。
    if module == 'geoloc':
        assert os.environ.get('HF_HUB_OFFLINE') == '1'
        assert os.environ.get('HF_HUB_CACHE') == str(RUNTIME / 'pretrained/huggingface/hub')
    else:
        assert os.environ.get('KERAS_HOME') == str(RUNTIME / 'pretrained/keras')
    output = args.output.resolve()
    # 强制所有写入位于独立 smoke 目录；拒绝已有输出，禁止覆盖正式运行目录。
    assert (RUNTIME / 'smoke-tests').resolve() in output.parents, output
    assert not output.exists(), f'拒绝覆盖已有输出：{output}'
    output.mkdir(parents=True)
    report = Report(module, output)
    protected = {}
    try:
        with report.stage('verify_pretrained_readonly'):
            for name in ('cvusa', 'convnext') if module == 'geoloc' else ('gcvit',):
                path, expected = ASSETS[name]
                assert digest(path) == expected, f'预训练文件校验失败：{path}'
                protected[str(path)] = expected
        work(args.data_root.resolve(), report, protected)
        with report.stage('verify_inputs_unchanged'):
            for path, expected in protected.items():
                assert digest(path) == expected, f'输入文件内容发生变化：{path}'
        report.state['status'] = 'passed'
    except BaseException:
        report.state['status'] = 'failed'
        report.state['traceback'] = traceback.format_exc()
        raise
    finally:
        report.save()
    print(module.upper() + '_SMOKE_PASSED', flush=True)
