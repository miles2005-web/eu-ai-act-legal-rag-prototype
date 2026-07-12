"""Build a separate metadata-v2 candidate corpus for the EU Data Act."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_act_corpus import (  # noqa: E402
    build_data_act_candidate_records,
    write_data_act_candidate_corpus,
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
    args = parser.parse_args()

    records = build_data_act_candidate_records(args.source)
    output = write_data_act_candidate_corpus(records, args.output)
    print(f"Data Act candidate records: {len(records)}")
    print(f"Candidate corpus: {output}")


if __name__ == "__main__":
    main()
