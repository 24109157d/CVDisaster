import argparse
from pathlib import Path
import subprocess
import sys

from common import (ROOT, check_manifest_settings, configuration, load_manifest, new_directory,
                    read_json, require_compute_node, write_json)
from analysis import primary_comparisons


def execute(script, *arguments):
    subprocess.run([sys.executable, '-u', str(ROOT / script), *map(str, arguments)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description='ConvNeXt-Tiny 新融合：独立验证、重训及贡献评价')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, choices=(0, 2), default=2)
    parser.add_argument('--geo-predictions', type=Path)
    args = parser.parse_args()
    require_compute_node()
    settings = configuration(args.config)
    check_manifest_settings(settings, load_manifest(args.manifest))
    output = new_directory(args.output)
    seed = settings['seed']
    variant = 'full'
    directory = new_directory(output / variant / str(seed))
    config = directory / 'config.json'
    write_json(config, settings)
    selected = directory / 'select'
    execute('train.py', '--config', config, '--manifest', args.manifest, '--output', selected,
            '--variant', variant, '--seed', seed, '--stage', 'select', '--gpus', args.gpus)
    refit = directory / 'refit'
    execute('train.py', '--config', config, '--manifest', args.manifest, '--output', refit,
            '--variant', variant, '--seed', seed, '--stage', 'refit', '--gpus', args.gpus,
            '--selection-run', selected)
    evaluation = directory / 'evaluation'
    extra = ['--geo-predictions', args.geo_predictions] if args.geo_predictions else []
    execute('evaluate.py', '--run', refit, '--manifest', args.manifest, '--output', evaluation,
            '--gpus', args.gpus, *extra)
    runs = {variant: [str(evaluation / 'predictions.csv')]}
    write_json(output / 'runs.json', runs)
    specification = output / 'comparisons.json'
    write_json(specification, {'comparisons': primary_comparisons(runs)})
    execute('analysis.py', '--comparisons', specification, '--output', output / 'statistics',
            '--group', 'spatial_group' if read_json(args.manifest / 'manifest.json')[
                'spatial_independence'] else 'sequence',
            '--repetitions', settings['bootstrap_repetitions'])


if __name__ == '__main__':
    main()
