import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_queries(path):
    frame = pd.read_csv(path, dtype={"query_id": "int64", "correct_rank": "int64"})
    if len(frame) != 1649 or frame.query_id.duplicated().any():
        raise ValueError(f"逐查询记录必须包含 1649 个唯一 ID: {path}")
    if not frame.correct_rank.between(0, 1648).all():
        raise ValueError(f"逐查询排名超出完整候选库范围: {path}")
    return frame.set_index("query_id").sort_index()


def query_metrics(frame):
    return {name: float((frame.correct_rank < cutoff).mean() * 100)
            for name, cutoff in {"R@1": 1, "R@5": 5, "R@10": 10, "R@1%": 16}.items()}


def compare_results(run, baseline_queries, output):
    run = Path(run)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    reference = metadata["config"]["baseline_reference"]
    if (result["architecture"], result["seed"], result["scheme"]) != ("geo_cvft_refine", 42, "geo_cvft_refine"):
        raise ValueError("本次比较要求 geo_cvft_refine、seed=42 的正式训练结果")
    if reference["job_id"] != "3979539":
        raise ValueError("本次比较要求使用作业 3979539 的已完成基线")
    improved = read_queries(run / "best_queries.csv")
    baseline = read_queries(baseline_queries)
    if not improved.index.equals(baseline.index):
        raise ValueError("改进模型和已有基线的完整测试 ID 不一致")
    measured = query_metrics(improved)
    baseline_measured = query_metrics(baseline)
    for name in measured:
        if not np.isclose(measured[name], result["metrics"][name], atol=1e-10, rtol=0):
            raise ValueError(f"改进模型指标与逐查询排名不一致: {name}")
        # 基线日志保留四位小数，逐查询统计使用完整精度。
        if abs(baseline_measured[name] - reference["metrics"][name]) > 5e-5:
            raise ValueError(f"基线逐查询记录与作业 3979539 的成绩不一致: {name}")
    first = baseline.correct_rank < 1
    second = improved.correct_rank < 1
    summary = {"scheme": result["scheme"], "seed": 42, "baseline_job_id": reference["job_id"],
               "baseline_training_commit": reference["training_commit"],
               "baseline_checkpoint": reference["checkpoint"], "fusion": result["fusion"],
               "metrics": {name: {"baseline": reference["metrics"][name],
                                   "geo_cvft_refine": result["metrics"][name],
                                   "difference_percentage_points": result["metrics"][name] - reference["metrics"][name]}
                           for name in measured},
               "baseline_recomputed_metrics": baseline_measured,
               "both_correct": int((first & second).sum()),
               "both_incorrect": int((~first & ~second).sum()),
               "baseline_incorrect_combined_correct": int((~first & second).sum()),
               "baseline_correct_combined_incorrect": int((first & ~second).sum()),
               "baseline_queries": str(Path(baseline_queries).resolve()),
               "improved_queries": str((run / "best_queries.csv").resolve())}
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repository):
        raise ValueError("比较结果必须保存在仓库之外")
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame({"query_id": baseline.index, "baseline_rank": baseline.correct_rank.to_numpy(),
                  "geo_cvft_refine_rank": improved.correct_rank.to_numpy(),
                  "baseline_correct": first.to_numpy(), "geo_cvft_refine_correct": second.to_numpy()}).to_csv(
                      output / "query_comparison.csv", index=False)
    return summary
