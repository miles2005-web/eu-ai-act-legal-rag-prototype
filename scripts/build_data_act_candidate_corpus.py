"""Build a separate metadata-v2 candidate corpus for the EU Data Act."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_act_corpus import (  # noqa: E402
    DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
    build_data_act_candidate_records,
    compare_data_act_candidate_to_runtime_pack,
    write_data_act_candidate_corpus,
    write_data_act_relevance_runtime_pack,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated metadata-v2 candidate corpus from an official "
            "Regulation (EU) 2023/2854 source."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Prepared Data Act .txt, text-based .pdf, or EUR-Lex .html source",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Candidate JSON path; must not be vector_store.json",
    )
    parser.add_argument(
        "--runtime-pack",
        type=Path,
        default=DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
        help="Committed Data Act runtime Evidence pack used for drift validation",
    )
    parser.add_argument(
        "--write-runtime-pack",
        action="store_true",
        help=(
            "Explicitly regenerate the reviewed runtime pack before drift "
            "validation; normal builds never rewrite it"
        ),
    )
    args = parser.parse_args()

    records = build_data_act_candidate_records(args.source)
    output = write_data_act_candidate_corpus(records, args.output)
    if args.write_runtime_pack:
        write_data_act_relevance_runtime_pack(records, args.runtime_pack)
    compare_data_act_candidate_to_runtime_pack(records, args.runtime_pack)
    print(f"Data Act candidate records: {len(records)}")
    print(f"Candidate corpus: {output}")
    print(f"Runtime pack verified: {args.runtime_pack}")
    print("Promotion status: candidate only; active vector store unchanged")


if __name__ == "__main__":
    main()
