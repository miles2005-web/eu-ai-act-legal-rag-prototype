"""Build the isolated AI Act Article 6(1) metadata-v2 candidate corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_act_product_safety_corpus import (  # noqa: E402
    DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
    build_ai_act_product_safety_candidate_records,
    compare_candidate_to_runtime_pack,
    write_ai_act_product_safety_candidate_corpus,
    write_ai_act_product_safety_runtime_pack,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated metadata-v2 candidate corpus for the EU AI "
            "Act Article 6(1) product-safety route."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Official English EUR-Lex HTML for Regulation (EU) 2024/1689",
    )
    parser.add_argument(
        "--runtime-pack",
        type=Path,
        default=DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
        help="Committed runtime Evidence pack used for drift validation",
    )
    parser.add_argument(
        "--write-runtime-pack",
        action="store_true",
        help=(
            "Explicitly regenerate the reviewed runtime pack before drift "
            "validation; normal builds never rewrite it"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Candidate JSON path; must not be vector_store.json",
    )
    args = parser.parse_args()

    records = build_ai_act_product_safety_candidate_records(args.source)
    output = write_ai_act_product_safety_candidate_corpus(records, args.output)
    if args.write_runtime_pack:
        write_ai_act_product_safety_runtime_pack(records, args.runtime_pack)
    compare_candidate_to_runtime_pack(records, args.runtime_pack)
    print(f"AI Act product-safety atomic records: {len(records)}")
    print(f"Candidate corpus: {output}")
    print(f"Runtime pack verified: {args.runtime_pack}")
    print("Promotion status: candidate only; active vector store unchanged")


if __name__ == "__main__":
    main()
