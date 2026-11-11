from types import SimpleNamespace

import pandas as pd
import torch
from torch.cuda.amp import autocast

from sample4geo.evaluate.cvusa_and_cvact import calculate_scores as author_calculate_scores
from sample4geo.model import TimmModel
from sample4geo.trainer import predict as author_predict
from .evaluation import rank_similarities
from .runtime import parallel_model, sha256, write_json


def validate_author_evaluation(checkpoint_path, loaders, common_result, config, output,
                               checkpoint_format="experiment"):
    author_model = TimmModel(config["backbone"], pretrained=False, img_size=384)
    saved = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_format == "author":
        state = saved
    elif checkpoint_format == "experiment":
        state = saved["model"]
    else:
        raise ValueError("checkpoint_format 必须为 author 或 experiment")
    author_model.load_state_dict(state, strict=True)
    del state
    del saved
    raw_dtypes = []
    handle = author_model.register_forward_hook(lambda module, inputs, result: raw_dtypes.append(str(result.dtype)))
    author_model = parallel_model(author_model)
    author_config = SimpleNamespace(device="cuda", verbose=False, normalize_features=True)
    # 直接执行作者的 predict 和 calculate_scores，保留其 AMP 与归一化顺序。
    reference, reference_ids = author_predict(author_config, author_model, loaders["test_satellite"])
    query, query_ids = author_predict(author_config, author_model, loaders["test_ground"])
    handle.remove()
    query_order_equal = torch.equal(query_ids.cpu(), common_result["ground_ids"])
    reference_order_equal = torch.equal(reference_ids.cpu(), common_result["satellite_ids"])
    report = {"checkpoint_sha256": sha256(checkpoint_path),
              "query_order_equal": query_order_equal, "reference_order_equal": reference_order_equal,
              "author_raw_dtypes": sorted(set(raw_dtypes)),
              "author_query_dtype": str(query.dtype), "author_reference_dtype": str(reference.dtype),
              "common_query_dtype": str(common_result["ground"]["descriptors"].dtype),
              "common_reference_dtype": str(common_result["satellite"]["descriptors"].dtype)}
    if not query_order_equal or not reference_order_equal:
        write_json(output / "author_evaluation_consistency.json", report)
        raise AssertionError("作者与共用评价流程的查询或候选 ID 顺序不一致")
    with autocast(enabled=False):
        cutoffs = {"R@1": 1, "R@5": 5, "R@10": 10, "R@1%": len(reference_ids) // 100}
        author_metrics = {name: float(author_calculate_scores(
            query, reference, query_ids, reference_ids, step_size=1000, ranks=[cutoff]))
                          for name, cutoff in cutoffs.items()}
        scores = torch.cat([(query[start:start + 1000] @ reference.T).cpu()
                            for start in range(0, len(query), 1000)])
    metrics, ranks = rank_similarities(scores, query_ids.cpu(), reference_ids.cpu())
    differences = (scores - common_result["scores"]).abs()
    rank_equal = torch.equal(ranks, common_result["ranks"])
    recall_equal = all(abs(author_metrics[name] - common_result["metrics"][name]) <= 1e-12 for name in cutoffs)
    author_rank_metrics_equal = all(abs(author_metrics[name] - metrics[name]) <= 1e-12 for name in cutoffs)
    report.update(author_similarity_dtype=str(scores.dtype), common_similarity_dtype=str(common_result["scores"].dtype),
                  max_absolute_similarity_error=float(differences.max()),
                  similarity_tolerance=config["author_similarity_tolerance"],
                  author_metrics=author_metrics, common_metrics=common_result["metrics"],
                  author_rank_metrics_equal=author_rank_metrics_equal,
                  ranks_equal=rank_equal, recalls_equal=recall_equal,
                  differing_query_ranks=int((ranks != common_result["ranks"]).sum()))
    report["passed"] = (report["max_absolute_similarity_error"] <= config["author_similarity_tolerance"]
                        and rank_equal and recall_equal and author_rank_metrics_equal)
    pd.DataFrame({"query_id": query_ids.cpu().tolist(), "author_rank": ranks.tolist(),
                  "common_rank": common_result["ranks"].tolist(),
                  "max_absolute_similarity_error": differences.amax(1).tolist()}).to_csv(
                      output / "author_evaluation_queries.csv", index=False)
    write_json(output / "author_evaluation_consistency.json", report)
    if not report["passed"]:
        raise AssertionError("作者评价一致性检查失败，数值、dtype 和逐查询排名已保存")
    return report
