"""将现有 GCViT 权重映射到 Keras 2.10 get_file 实际使用的默认缓存位置。"""

import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCE = Path('/scratch/liuzeyu/heyinxi/pretrained/keras/datasets/gcvit_tiny_weights.h5')
EXPECTED_SHA256 = '90bf9bfaafd086a831e92edbbfbba9450fd56786c2d39b3ac3f215d0cef54768'


def sha256(path):
    """顺序读取权重进行完整性校验，不写入或复制权重内容。"""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(check_only=False):
    source = SOURCE.resolve(strict=True)
    assert source.is_file() and sha256(source) == EXPECTED_SHA256, '现有 GCViT 权重校验失败'
    base = Path.home() / '.keras'
    target = base / 'datasets' / SOURCE.name
    action = 'checked'
    if check_only:
        # 训练作业只检查映射，不创建目录、不覆盖文件、不尝试下载。
        assert base.is_dir() and os.access(base, os.W_OK), '默认缓存目录不可写，Keras 会改用 /tmp'
        assert target.is_symlink() and target.resolve(strict=True) == source, f'缓存映射未准备：{target}'
    else:
        # 仅创建缺失的精确文件软链接；遇到其他内容直接报错，由用户核对后处理。
        base.mkdir(parents=True, exist_ok=True)
        assert os.access(base, os.W_OK), '默认缓存目录不可写'
        target.parent.mkdir(exist_ok=True)
        if target.is_symlink():
            assert target.resolve(strict=True) == source, f'已有软链接指向其他文件，拒绝覆盖：{target}'
            action = 'already_linked'
        elif target.exists():
            raise FileExistsError(f'已有普通文件或目录，拒绝覆盖：{target}')
        else:
            target.symlink_to(source)
            action = 'created_link'
    assert target.samefile(source) and sha256(target) == EXPECTED_SHA256
    result = {'action': action, 'source': str(source), 'keras_default_cache': str(target),
              'sha256': EXPECTED_SHA256, 'same_file': True}
    print('EST_CACHE_READY ' + json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args().check_only)
