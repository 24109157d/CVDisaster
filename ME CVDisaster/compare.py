import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="比较两个模型在三个相同 seed 下的 best 结果")
    parser.add_argument("--runs", nargs=6, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        raise ValueError("比较结果必须保存在仓库之外")
    runs = {}
    for directory in map(Path, args.runs):
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        key = (result["architecture"], result["seed"])
        if key in runs:
            raise ValueError(f"重复的模型与随机种子: {key}")
        queries = pd.read_csv(directory / "best_queries.csv", dtype={"query_id": "int64"})
        if len(queries) != 1649 or queries.query_id.duplicated().any():
            raise ValueError(f"逐查询结果的数量或唯一性错误: {directory}")
        if result["scheme"] != "geo_cvft_refine":
            raise ValueError(f"方案编号不匹配: {directory}")
        if result["architecture"] == "baseline":
            consistency = result["author_evaluation_consistency"]
            if not consistency["passed"] or consistency["checkpoint_sha256"] != result["feature_context"]["checkpoint_id"]:
                raise ValueError(f"baseline checkpoint 尚未通过作者评价一致性核验: {directory}")
        cutoffs = {"R@1": 1, "R@5": 5, "R@10": 10, "R@1%": 16}
        for metric, cutoff in cutoffs.items():
            measured = float((queries.correct_rank < cutoff).mean() * 100)
            if not np.isclose(measured, result["metrics"][metric], atol=1e-10, rtol=0):
                raise ValueError(f"指标与逐查询结果不一致: {directory} {metric}")
        run_metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        runs[key] = (result, queries.set_index("query_id").sort_index(), run_metadata)
    architectures = ("baseline", "geo_cvft_refine")
    seeds = (42, 43, 44)
    if set(runs) != {(architecture, seed) for architecture in architectures for seed in seeds}:
        raise ValueError("必须提供两个模型分别使用 seed 42、43、44 的六次完整结果")
    rows = []
    common_ids = None
    common_config = None
    common_manifest = None
    for seed in seeds:
        baseline, baseline_queries, baseline_metadata = runs[("baseline", seed)]
        combined, combined_queries, combined_metadata = runs[("geo_cvft_refine", seed)]
        if baseline_metadata["split_manifest"] != combined_metadata["split_manifest"]:
            raise ValueError(f"seed {seed} 的两种模型使用了不同数据划分")
        for metadata in (baseline_metadata, combined_metadata):
            if common_manifest is None:
                common_manifest = metadata["split_manifest"]
            elif common_manifest != metadata["split_manifest"]:
                raise ValueError("六次实验的数据划分文件不同")
            if common_config is None:
                common_config = metadata["config"]
            elif common_config != metadata["config"]:
                raise ValueError("六次实验的公共配置不同")
        if not baseline_queries.index.equals(combined_queries.index):
            raise ValueError(f"seed {seed} 的两种模型使用了不同测试 ID")
        if common_ids is None:
            common_ids = baseline_queries.index
        elif not common_ids.equals(baseline_queries.index):
            raise ValueError("三个随机种子的测试 ID 不一致")
        first = baseline_queries.correct_rank < 1
        second = combined_queries.correct_rank < 1
        rows.append({"seed": seed, "baseline_R@1": baseline["metrics"]["R@1"],
                     "geo_cvft_refine_R@1": combined["metrics"]["R@1"],
                     "fusion_logit": combined["fusion"]["fusion_logit"],
                     "fusion_weight": combined["fusion"]["fusion_weight"],
                     "difference_percentage_points": combined["metrics"]["R@1"] - baseline["metrics"]["R@1"],
                     "both_correct": int((first & second).sum()),
                     "both_incorrect": int((~first & ~second).sum()),
                     "baseline_incorrect_combined_correct": int((~first & second).sum()),
                     "baseline_correct_combined_incorrect": int((first & ~second).sum())})
    summary = {}
    for architecture in architectures:
        summary[architecture] = {}
        for metric in ("R@1", "R@5", "R@10", "R@1%"):
            values = [runs[(architecture, seed)][0]["metrics"][metric] for seed in seeds]
            summary[architecture][metric] = {"values": values, "mean": float(np.mean(values)),
                                             "sample_std": float(np.std(values, ddof=1))}
    summary["paired_comparisons"] = rows
    summary["historical_reference"] = {"job_id": "3979539", "R@1": 79.8059,
                                       "R@5": 97.8169, "R@10": 98.9691, "R@1%": 99.5149}
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "paired_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
