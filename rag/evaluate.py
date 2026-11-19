"""检索层评测：python -m rag.evaluate --output retrieval_results.json。"""
import argparse
import json
import time
from pathlib import Path

from rag.evidence import source_label


DEFAULT_DATASET = Path(__file__).with_name("eval_cases.jsonl")


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            if (not isinstance(case, dict)
                    or not isinstance(case.get("id"), str)
                    or not isinstance(case.get("query"), str)
                    or not case["query"].strip()
                    or not isinstance(case.get("expected_sources"), list)
                    or not case["expected_sources"]
                    or not all(isinstance(s, str) and s for s in case["expected_sources"])):
                raise ValueError(f"第 {number} 行评测数据格式无效")
            if case["id"] in seen:
                raise ValueError(f"第 {number} 行题目 ID 重复：{case['id']}")
            seen.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError("评测集为空")
    return cases


def score_sources(retrieved: list[str], expected: list[str]) -> dict:
    """只评估预期文件是否被召回；不评估回答质量或事实准确性。"""
    rank = next((index for index, source in enumerate(retrieved, 1)
                 if source in expected), None)
    return {"hit": rank is not None, "rank": rank, "reciprocal_rank": 1 / rank if rank else 0.0}


def evaluate(cases: list[dict], retrieve, k: int) -> dict:
    if k < 1:
        raise ValueError("k 必须为正整数")
    results = []
    for case in cases:
        start = time.perf_counter()
        docs = retrieve(case["query"], k)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        sources = [source_label(doc.metadata) for doc in docs]
        score = score_sources(sources, case["expected_sources"])
        results.append({
            "id": case["id"], "query": case["query"],
            "expected_sources": case["expected_sources"],
            "retrieved_sources": sources, "latency_ms": elapsed_ms, **score,
        })
    return {
        "count": len(results), "k": k,
        "hit_at_k": sum(item["hit"] for item in results) / len(results),
        "mrr_at_k": sum(item["reciprocal_rank"] for item in results) / len(results),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description="按知识库来源文件评估检索质量")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, help="可选：将逐题结果保存为 JSON")
    args = parser.parse_args()
    cases = load_cases(args.dataset)

    # 延迟导入：纯指标计算及单元测试无需 API Key 或 Chroma 依赖。
    from rag.vector_store import VectorStoreService

    store = VectorStoreService().vector_store

    def retrieve(query: str, k: int):
        return store.similarity_search(query, k=k)

    report = evaluate(cases, retrieve, args.k)
    print(f"题数：{report['count']}，Hit@{args.k}：{report['hit_at_k']:.2%}，"
          f"MRR@{args.k}：{report['mrr_at_k']:.3f}")
    for item in report["cases"]:
        print(f"{item['id']}: {'命中' if item['hit'] else '未命中'} "
              f"(排名 {item['rank']}) | 召回：{', '.join(item['retrieved_sources'])}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"逐题结果已保存：{args.output}")


if __name__ == "__main__":
    main()
