import argparse
import ast
import importlib.metadata
import os
from pathlib import Path

from common import DEFAULT_CONFIG, read_config, sha256


def main():
    parser = argparse.ArgumentParser(description='检查串联作业所用环境、源码及现有初始权重')
    parser.add_argument('--stage', choices=['geo', 'est'], required=True)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = read_config(args.config)
    packages = ({'torch': '1.13.1+cu116', 'torchvision': '0.14.1+cu116', 'timm': '0.8.19.dev0'}
                if args.stage == 'geo' else
                {'tensorflow': '2.10.1', 'keras': '2.10.0', 'gcvit': '1.1.6', 'numpy': '1.23.5'})
    versions = {name: importlib.metadata.version(name) for name in packages}
    if versions != packages:
        raise ValueError(f'环境版本与复现环境不同: {versions}')
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    if args.stage == 'geo':
        params = config['geo']
        if sha256(params['initial_checkpoint']) != params['initial_sha256']:
            raise ValueError('CVUSA初始权重SHA256不同')
        cached_model = Path(os.environ['HF_HUB_CACHE']) / (
            'models--timm--convnext_base.fb_in22k_ft_in1k_384/snapshots/'
            '2b2698d86523262ca8065d0957c7f80d963bb43f/model.safetensors')
        if not cached_model.is_file() or cached_model.stat().st_size == 0:
            raise FileNotFoundError(cached_model)
    else:
        source = Path(config['est']['initial_weights']).resolve(strict=True)
        target = Path.home() / '.keras' / 'datasets' / source.name
        if (sha256(source) != '90bf9bfaafd086a831e92edbbfbba9450fd56786c2d39b3ac3f215d0cef54768'
                or target.resolve(strict=True) != source
                or not os.access(target.parent.parent, os.W_OK)):
            raise ValueError('GCViT预训练权重或Keras默认缓存映射不同')
    print('COMBINED_ENVIRONMENT_OK', args.stage, versions, flush=True)


if __name__ == '__main__':
    main()
