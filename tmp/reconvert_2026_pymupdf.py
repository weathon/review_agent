"""
Re-convert existing ICLR 2026 PDFs using the old pymupdf4llm parser.

Reads PDFs from iclr2026_data/pdfs/ and ratings from iclr2026_data/ratings.csv,
writes converted text to iclr2026_data_pymupdf/papers/ and copies ratings.csv.
"""

import csv
import re
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_PDF_DIR = Path(__file__).resolve().parent.parent / "iclr2026_data" / "pdfs"
SRC_RATINGS = Path(__file__).resolve().parent.parent / "iclr2026_data" / "ratings.csv"

OUT_DIR = Path(__file__).resolve().parent.parent / "iclr2026_data_pymupdf"
OUT_PAPERS = OUT_DIR / "papers"
OUT_RATINGS = OUT_DIR / "ratings.csv"


def pdf_to_markdown(pdf_path: Path) -> str:
    """Convert PDF to markdown text using pymupdf4llm (old method)."""
    import pymupdf4llm
    try:
        text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        print(f"    PDF conversion error: {e}")
        # Fallback to pymupdf
        try:
            import pymupdf
            doc = pymupdf.open(str(pdf_path))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")
            return ""

    # Clean up line numbers (e.g. **000**, **001**, **012 013**)
    text = re.sub(r"\*\*\d{3}(?:\s+\d{3})*\*\*\s*", "", text)
    # Remove status headers that leak review outcome
    text = re.sub(r"Under review as a conference paper at ICLR \d{4}\s*\n?", "", text)
    text = re.sub(r"Published as a conference paper at ICLR \d{4}\s*\n?", "", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def convert_one(pdf_path: Path) -> tuple[str, bool]:
    paper_id = pdf_path.stem
    out_path = OUT_PAPERS / f"{paper_id}.txt"
    if out_path.exists() and out_path.stat().st_size > 100:
        return paper_id, True

    text = pdf_to_markdown(pdf_path)
    if not text or len(text) < 500:
        return paper_id, False

    out_path.write_text(text, encoding="utf-8")
    return paper_id, True


def main():
    OUT_DIR.mkdir(exist_ok=True)
    OUT_PAPERS.mkdir(exist_ok=True)

    # Copy ratings as-is
    shutil.copy2(SRC_RATINGS, OUT_RATINGS)
    print(f"Copied ratings to {OUT_RATINGS}")

    # Get list of paper IDs from ratings
    with open(SRC_RATINGS) as f:
        reader = csv.DictReader(f)
        paper_ids = {row["paper_id"] for row in reader}
    print(f"{len(paper_ids)} papers in ratings")

    # Find PDFs
    pdfs = [SRC_PDF_DIR / f"{pid}.pdf" for pid in paper_ids if (SRC_PDF_DIR / f"{pid}.pdf").exists()]
    print(f"{len(pdfs)} PDFs found, converting with pymupdf4llm...")

    success = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(convert_one, pdf): pdf for pdf in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, ok = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  FAILED: {pid}")
            if i % 20 == 0:
                print(f"  {i}/{len(pdfs)} done...")

    print(f"\nDone! {success} converted, {failed} failed.")
    print(f"Output: {OUT_PAPERS}")


if __name__ == "__main__":
    main()
