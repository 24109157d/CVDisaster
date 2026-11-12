import json
from pathlib import Path

import pandas as pd


def read_queries(path):
    frame = pd.read_csv(path, dtype={"query_id": "int64", "correct_rank": "int64"})
    return frame.set_index("query_id").sort_index()


def query_metrics(frame):
    return {name: float((frame.correct_rank < cutoff).mean() * 100)
            for name, cutoff in {"R@1": 1, "R@5": 5, "R@10": 10, "R@1%": len(frame) // 100}.items()}


def compare_results(run, baseline_queries, output=None):
    run = Path(run)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    reference = metadata["config"]["baseline_reference"]
    improved = read_queries(run / "best_queries.csv")
    baseline = read_queries(baseline_queries)
    if not improved.index.equals(baseline.index):
        raise ValueError("改进模型和已有基线的完整测试 ID 不一致")
    measured = query_metrics(improved)
    baseline_measured = query_metrics(baseline)
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
    if output is None:
        return summary
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
