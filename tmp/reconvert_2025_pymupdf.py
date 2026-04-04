"""
Re-convert ICLR 2025 PDFs using current pymupdf4llm (0.3.4 + PyMuPDF 1.27).
Reads PDFs from iclr2025_data/pdfs/, writes to iclr2025_data_v2/papers/.
Copies ratings from iclr2025_data/ratings.csv.
"""

import csv
import re
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_PDF_DIR = Path(__file__).resolve().parent.parent / "iclr2025_data" / "pdfs"
SRC_RATINGS = Path(__file__).resolve().parent.parent / "iclr2025_data" / "ratings.csv"

OUT_DIR = Path(__file__).resolve().parent.parent / "iclr2025_data_v2"
OUT_PAPERS = OUT_DIR / "papers"
OUT_RATINGS = OUT_DIR / "ratings.csv"


def pdf_to_markdown(pdf_path: Path) -> str:
    import pymupdf4llm
    try:
        text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        print(f"    PDF conversion error: {e}")
        try:
            import pymupdf
            doc = pymupdf.open(str(pdf_path))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")
            return ""

    text = re.sub(r"\*\*\d{3}(?:\s+\d{3})*\*\*\s*", "", text)
    text = re.sub(r"Under review as a conference paper at ICLR \d{4}\s*\n?", "", text)
    text = re.sub(r"Published as a conference paper at ICLR \d{4}\s*\n?", "", text)
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

    shutil.copy2(SRC_RATINGS, OUT_RATINGS)
    print(f"Copied ratings to {OUT_RATINGS}")

    with open(SRC_RATINGS) as f:
        paper_ids = {row["paper_id"] for row in csv.DictReader(f)}
    print(f"{len(paper_ids)} papers in ratings")

    pdfs = [SRC_PDF_DIR / f"{pid}.pdf" for pid in paper_ids if (SRC_PDF_DIR / f"{pid}.pdf").exists()]
    print(f"{len(pdfs)} PDFs found, converting with pymupdf4llm...")

    success = failed = 0
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
