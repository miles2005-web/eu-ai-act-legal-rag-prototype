from __future__ import annotations

import argparse
from pathlib import Path

from src.legal_chunks import clean_extracted_text


RAW_DIR = Path("data/raw")
PARSED_DIR = Path("data/parsed")


def read_txt_file(path: Path) -> str:
    """Read a plain text file as UTF-8."""
    return path.read_text(encoding="utf-8").strip()


def read_pdf_file(path: Path) -> str:
    """
    Read a PDF file with pypdf if it is installed.

    This is intentionally basic. It extracts text page by page and joins it.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires `pypdf`. Install project requirements first."
        ) from exc

    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    return clean_extracted_text("\n\n".join(pages))


def parse_document(path: Path) -> str:
    """Route a file to the right basic parser based on its extension."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return clean_extracted_text(read_txt_file(path))
    if suffix == ".pdf":
        return read_pdf_file(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def output_path_for(path: Path) -> Path:
    """Write every parsed document as a .txt file in data/parsed."""
    return PARSED_DIR / f"{path.stem}.txt"


def ingest(raw_dir: Path = RAW_DIR, parsed_dir: Path = PARSED_DIR) -> list[dict]:
    """Parse supported files from data/raw and save plain text into data/parsed."""
    parsed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    supported_paths = sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    )

    for path in supported_paths:
        try:
            text = parse_document(path)
            if not text:
                results.append({"source": path.name, "status": "skipped", "reason": "empty content"})
                continue

            destination = output_path_for(path)
            destination.write_text(text + "\n", encoding="utf-8")
            results.append({"source": path.name, "status": "parsed", "output": destination.name})
        except Exception as exc:
            results.append({"source": path.name, "status": "failed", "reason": str(exc)})

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse .txt and basic .pdf files from data/raw into data/parsed."
    )
    parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    results = ingest()

    if not results:
        print("No supported files found in data/raw.")
        return

    for item in results:
        if item["status"] == "parsed":
            print(f"[parsed] {item['source']} -> {item['output']}")
        elif item["status"] == "skipped":
            print(f"[skipped] {item['source']}: {item['reason']}")
        else:
            print(f"[failed] {item['source']}: {item['reason']}")


if __name__ == "__main__":
    main()
