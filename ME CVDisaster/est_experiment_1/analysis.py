import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import new_directory, read_json, write_json


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('p 值必须为 [0,1] 内的有限数值')
    order = np.argsort(values)
    corrected = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty_like(values)
    result[order] = np.minimum(corrected, 1)
    return result.tolist()


def net_benefit(base_correct, final_correct):
    base = np.asarray(base_correct)
    final = np.asarray(final_correct)
    if base.ndim != 1 or final.shape != base.shape or not len(base):
        raise ValueError('净收益统计需要同一批非空目标')
    if not np.isin(base, [0, 1]).all() or not np.isin(final, [0, 1]).all():
        raise ValueError('净收益统计要求逐目标的正确性记录')
    fixed = int(((base == 0) & (final == 1)).sum())
    corrupted = int(((base == 1) & (final == 0)).sum())
    return {'Nfixed': fixed, 'Ncorrupted': corrupted, 'delta_N': fixed - corrupted,
            'net_benefit_pp': 100.0 * (fixed - corrupted) / len(base)}


def load_predictions(paths, condition, subset, prediction_column='prediction'):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={'sample_id': str, 'sequence': str, 'spatial_group': str})
        selected = frame[frame['condition'] == condition].copy()
        selected['prediction'] = selected[prediction_column]
        if subset == 'geo_wrong':
            selected = selected[selected['geo_wrong'] == 1]
        if selected.empty:
            raise ValueError(f'没有对应评价记录: {path}, {condition}, {subset}')
        if selected.duplicated(['training_seed', 'sample_id', 'condition_seed']).any():
            raise ValueError('同一目标和输入随机种子出现重复预测')
        if not set(selected['label']).issubset({0, 1, 2}) or not set(selected['prediction']).issubset({0, 1, 2}):
            raise ValueError('评价记录包含无效类别')
        repeated = selected.groupby(['training_seed', 'sample_id'])['condition_seed'].agg(lambda values: tuple(sorted(values)))
        if repeated.nunique() != 1:
            raise ValueError('各目标的输入随机种子集合不一致')
        if selected.groupby(['training_seed', 'sample_id'])['condition_seed'].nunique().nunique() != 1:
            raise ValueError('同一条件的随机输入次数不一致')
        selected['correct'] = (selected['prediction'] == selected['label']).astype(float)
        keep = ['training_seed', 'sample_id', 'sequence', 'spatial_group', 'label']
        averaged = selected.groupby(keep, dropna=False)['correct'].mean().reset_index()
        # macro-F1 在每份真实随机输入结果上计算，再对输入随机性平均。
        frames.append((averaged, selected))
    combined = pd.concat([item[0] for item in frames], ignore_index=True)
    raw = pd.concat([item[1] for item in frames], ignore_index=True)
    if combined.duplicated(['training_seed', 'sample_id']).any():
        raise ValueError('同一个训练种子和目标出现重复评价文件')
    target_sets = combined.groupby('training_seed')['sample_id'].agg(lambda values: tuple(sorted(values)))
    if target_sets.nunique() != 1:
        raise ValueError('各训练种子的测试目标集合不一致')
    combined.attrs['raw'] = raw.to_dict('list')
    return combined


def confusion_groups(frame, seeds, groups, group_column):
    raw = pd.DataFrame(frame.attrs['raw'])
    repeats = sorted(raw['condition_seed'].unique())
    result = np.zeros((len(seeds), len(groups), len(repeats), 3, 3), dtype=np.int64)
    seed_index = {value: index for index, value in enumerate(seeds)}
    group_index = {value: index for index, value in enumerate(groups)}
    repeat_index = {value: index for index, value in enumerate(repeats)}
    indices = (raw['training_seed'].map(seed_index).to_numpy(),
               raw[group_column].map(group_index).to_numpy(),
               raw['condition_seed'].map(repeat_index).to_numpy(),
               raw['label'].to_numpy(), raw['prediction'].to_numpy())
    np.add.at(result, indices, 1)
    return result


def macro_f1(confusion):
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    scores = np.divide(2.0 * true_positive, denominator,
                       out=np.zeros_like(true_positive, dtype=float), where=denominator > 0)
    return float(scores.mean())


def subtract_scores(left, right):
    keys = ['training_seed', 'sample_id', 'sequence', 'spatial_group', 'label']
    result = left.merge(right, on=keys, suffixes=('_left', '_right'), validate='one_to_one', how='outer', indicator=True)
    if not (result['_merge'] == 'both').all():
        raise ValueError('卫星收益差值的目标或训练种子不一致')
    result['correct'] = result['correct_left'] - result['correct_right']
    return result[keys + ['correct']]


def paired_group_comparison(left, right, group_column, repetitions=10000, seed=42, contrast=False):
    if repetitions < 100:
        raise ValueError('统计重采样次数至少为 100')
    keys = ['training_seed', 'sample_id']
    joined = left.merge(right, on=keys, suffixes=('_left', '_right'), validate='one_to_one', how='outer', indicator=True)
    if not (joined['_merge'] == 'both').all():
        raise ValueError('配对比较的训练种子或目标 ID 集合不一致')
    for column in ('label', 'sequence', group_column):
        a = joined[f'{column}_left'].fillna('')
        b = joined[f'{column}_right'].fillna('')
        if not a.equals(b) or (column == group_column and (a == '').any()):
            raise ValueError(f'配对比较的 {column} 不一致或缺失')
    joined['delta'] = joined['correct_left'] - joined['correct_right']
    seeds = sorted(joined['training_seed'].unique())
    groups = sorted(joined[f'{group_column}_left'].unique())
    sums = np.zeros((len(seeds), len(groups)))
    counts = np.zeros_like(sums)
    for i, training_seed in enumerate(seeds):
        part = joined[joined['training_seed'] == training_seed]
        grouped = part.groupby(f'{group_column}_left')['delta'].agg(['sum', 'count'])
        if set(grouped.index) != set(groups):
            raise ValueError('各训练种子的独立分组集合不一致')
        sums[i] = grouped.loc[groups, 'sum']
        counts[i] = grouped.loc[groups, 'count']
    if len(groups) < 2:
        raise ValueError('独立分组不足，无法执行分组推断')
    rng = np.random.default_rng(seed)
    observed = float(np.mean(sums.sum(axis=1) / counts.sum(axis=1)))
    bootstrap = []
    left_confusion = confusion_groups(left, seeds, groups, group_column) if not contrast else None
    right_confusion = confusion_groups(right, seeds, groups, group_column) if not contrast else None
    f1_bootstrap = []
    for _ in range(repetitions):
        g = rng.integers(0, len(groups), len(groups))
        s = rng.integers(0, len(seeds), len(seeds))
        bootstrap.append(np.mean(sums[s][:, g].sum(axis=1) / counts[s][:, g].sum(axis=1)))
        if not contrast:
            f1_bootstrap.append(macro_f1(left_confusion[s][:, g].sum(axis=1))
                                - macro_f1(right_confusion[s][:, g].sum(axis=1)))
    exhaustive = len(groups) <= 16
    signs = itertools.product((-1, 1), repeat=len(groups)) if exhaustive else (
        rng.choice((-1, 1), len(groups)) for _ in range(repetitions))
    extreme = 0
    total = 0
    for sign in signs:
        permuted = np.mean((sums * np.asarray(sign)).sum(axis=1) / counts.sum(axis=1))
        extreme += abs(permuted) >= abs(observed) - 1e-12
        total += 1
    pvalue = extreme / total if exhaustive else (extreme + 1) / (total + 1)
    each_seed = []
    for training_seed in seeds:
        part = joined[joined['training_seed'] == training_seed]
        a = part['correct_left'].to_numpy()
        b = part['correct_right'].to_numpy()
        discrete = not contrast and np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
        good = int(((a == 1) & (b == 0)).sum()) if discrete else None
        bad = int(((a == 0) & (b == 1)).sum()) if discrete else None
        p = binomtest(good, good + bad).pvalue if discrete and good + bad else (1.0 if discrete else None)
        benefit = net_benefit(b, a) if discrete else {
            'Nfixed': None, 'Ncorrupted': None, 'delta_N': None, 'net_benefit_pp': None}
        each_seed.append({'seed': int(training_seed), 'left_score': float(a.mean()),
                          'right_score': float(b.mean()), 'gain_pp': float((a - b).mean() * 100),
                          'corrected_count': good, 'harmed_count': bad,
                          **benefit,
                          'exploratory_mcnemar_p': p})
    return {'gain_pp': observed * 100, 'ci95_pp': (np.quantile(bootstrap, [0.025, 0.975]) * 100).tolist(),
            'p_value': float(pvalue), 'groups': len(groups), 'training_seeds': len(seeds),
            'count_per_seed': len(joined) // len(seeds), 'per_seed': each_seed,
            'permutation_exhaustive': exhaustive, 'permutations': total,
            'comparison_type': 'difference_of_differences' if contrast else 'paired_difference',
            'macro_f1_gain': None if contrast else macro_f1(left_confusion.sum(axis=1)) - macro_f1(right_confusion.sum(axis=1)),
            'macro_f1_ci95': None if contrast else np.quantile(f1_bootstrap, [0.025, 0.975]).tolist()}


def primary_comparisons(runs):
    if set(runs) != {'full'} or not runs['full']:
        raise ValueError('当前实验需要 ConvNeXt-Tiny 新融合模型的预测')
    comparisons = [{'name': 'neighbor_net_benefit', 'family': 'neighbor',
                    'left': {'files': runs['full'], 'condition': 'paired'},
                    'right': {'files': runs['full'], 'condition': 'paired',
                              'prediction_column': 'base_prediction'}}]
    for condition in ('wrong', 'noise', 'gaussian', 'zero'):
        comparisons.append({'name': f'{condition}_gap', 'family': 'satellite_replacement',
                            'left': {'files': runs['full'], 'condition': 'paired'},
                            'right': {'files': runs['full'], 'condition': condition}})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description='EST 预设比较族的配对分组统计')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--comparisons', type=Path)
    source.add_argument('--run-index', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=10000)
    parser.add_argument('--group', choices=('sequence', 'spatial_group'), default='sequence')
    args = parser.parse_args()
    if args.comparisons is not None:
        specification = read_json(args.comparisons)
    else:
        runs = {}
        for index in args.run_index:
            for name, paths in read_json(index).items():
                runs.setdefault(name, []).extend(paths)
        specification = {'comparisons': primary_comparisons(runs)}
    results = []
    for item in specification['comparisons']:
        subset = item.get('subset', 'all')
        left = load_predictions(item['left']['files'], item['left']['condition'], subset,
                                item['left'].get('prediction_column', 'prediction'))
        right = load_predictions(item['right']['files'], item['right']['condition'], subset,
                                 item['right'].get('prediction_column', 'prediction'))
        if 'subtract' in item:
            sub_left = load_predictions(item['subtract']['left']['files'], item['subtract']['left']['condition'], subset,
                                        item['subtract']['left'].get('prediction_column', 'prediction'))
            sub_right = load_predictions(item['subtract']['right']['files'], item['subtract']['right']['condition'], subset,
                                         item['subtract']['right'].get('prediction_column', 'prediction'))
            left = subtract_scores(left, sub_left)
            right = subtract_scores(right, sub_right)
        result = paired_group_comparison(left, right, args.group, args.repetitions, contrast='subtract' in item)
        results.append({'name': item['name'], 'family': item['family'], **result})
    for family in sorted({item['family'] for item in results}):
        members = [item for item in results if item['family'] == family]
        for item, pvalue in zip(members, holm([item['p_value'] for item in members])):
            item['holm_p_value'] = pvalue
    output = new_directory(args.output)
    write_json(output / 'statistics.json', {'group': args.group, 'comparisons': results,
               'bootstrap_repetitions': args.repetitions,
               'confidence_intervals': '逐项 95% 区间；显著性判断使用同族 Holm 校正 p 值'})


if __name__ == '__main__':
    main()
