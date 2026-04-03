from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.legal_chunks import build_structured_chunks
from src.legal_chunks import score_chunk


PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
GOLDEN_QUERIES_PATH = PROJECT_ROOT / "eval" / "golden_queries.json"


def load_documents(data_dir: Path) -> list[dict]:
    documents = []

    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": path.name, "text": text})

    return documents


def build_index(documents: list[dict]) -> list[dict]:
    indexed_chunks = []

    for document in documents:
        indexed_chunks.extend(
            build_structured_chunks(
                text=document["text"],
                source=document["source"],
            )
        )

    return indexed_chunks


def retrieve(query: str, indexed_chunks: list[dict], top_k: int) -> list[dict]:
    scored_chunks = []

    for chunk in indexed_chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            scored_chunks.append({**chunk, "score": score})

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def load_golden_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def match_expectation(chunk: dict, expectation: dict) -> bool:
    return all(chunk.get(field) == value for field, value in expectation.items())


def evaluate_case(case: dict, indexed_chunks: list[dict], top_k: int) -> dict:
    results = retrieve(case["query"], indexed_chunks=indexed_chunks, top_k=top_k)
    top_1_hit = False
    top_k_hit = False

    if "expect" in case:
        top_1_hit = bool(results) and match_expectation(results[0], case["expect"])
        top_k_hit = any(match_expectation(result, case["expect"]) for result in results)
    elif "expect_any" in case:
        top_1_hit = bool(results) and any(
            match_expectation(results[0], expectation) for expectation in case["expect_any"]
        )
        top_k_hit = any(
            match_expectation(result, expectation)
            for result in results
            for expectation in case["expect_any"]
        )

    return {
        "id": case["id"],
        "query": case["query"],
        "top_1_hit": top_1_hit,
        "top_k_hit": top_k_hit,
        "top_result": results[0] if results else None,
    }


def format_result_line(result: dict) -> str:
    status = "PASS" if result["top_k_hit"] else "FAIL"
    top_result = result["top_result"]

    if not top_result:
        return f"[{status}] {result['id']}: no matches returned"

    labels = []
    if top_result.get("chapter_heading"):
        labels.append(top_result["chapter_heading"])
    if top_result.get("section_heading"):
        labels.append(top_result["section_heading"])
    if top_result.get("article_number"):
        labels.append(f"Article {top_result['article_number']}")
    if top_result.get("annex_ref"):
        labels.append(f"Annex {top_result['annex_ref']}")
    if top_result.get("recital_ref"):
        labels.append(f"Recital {top_result['recital_ref']}")

    metadata_label = " | ".join(labels) if labels else "no metadata"
    return (
        f"[{status}] {result['id']}: top1={'yes' if result['top_1_hit'] else 'no'}, "
        f"topk={'yes' if result['top_k_hit'] else 'no'} -> {metadata_label}"
    )


def main() -> None:
    documents = load_documents(PARSED_DIR)
    if not documents:
        raise SystemExit("No parsed documents found in data/parsed.")

    indexed_chunks = build_index(documents)
    golden_queries = load_golden_queries(GOLDEN_QUERIES_PATH)

    results = [evaluate_case(case, indexed_chunks=indexed_chunks, top_k=3) for case in golden_queries]
    top_1_hits = sum(1 for result in results if result["top_1_hit"])
    top_k_hits = sum(1 for result in results if result["top_k_hit"])

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(indexed_chunks)}")
    print(f"Queries: {len(results)}")
    print(f"Top-1 accuracy: {top_1_hits}/{len(results)}")
    print(f"Top-3 accuracy: {top_k_hits}/{len(results)}")
    print("")

    for result in results:
        print(format_result_line(result))


if __name__ == "__main__":
    main()
