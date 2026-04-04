"""
Re-parse the exact same papers from iclr2025_data.old using current pymupdf4llm.
Downloads PDFs for those specific paper IDs, converts with current parser,
copies ratings.csv as-is.
"""

import csv
import os
import re
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

OLD_DIR = Path(__file__).resolve().parent.parent / "iclr2025_data.old"
OUT_DIR = Path(__file__).resolve().parent.parent / "iclr2025_data_v3"
OUT_PAPERS = OUT_DIR / "papers"
OUT_PDFS = OUT_DIR / "pdfs"
OUT_RATINGS = OUT_DIR / "ratings.csv"


def get_or_client():
    import openreview
    return openreview.api.OpenReviewClient(
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ["OPENREVIEW_PASSWORD"],
        baseurl="https://api2.openreview.net",
    )


def download_pdf(client, paper_id, outfile):
    if outfile.exists() and outfile.stat().st_size > 0:
        return True
    try:
        pdf_bytes = client.get_pdf(paper_id)
        if len(pdf_bytes) > 1000:
            outfile.write_bytes(pdf_bytes)
            return True
    except Exception as e:
        print(f"  Download error {paper_id}: {e}")
    return False


def pdf_to_markdown(pdf_path):
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


def main():
    OUT_DIR.mkdir(exist_ok=True)
    OUT_PAPERS.mkdir(exist_ok=True)
    OUT_PDFS.mkdir(exist_ok=True)

    # Copy ratings as-is
    shutil.copy2(OLD_DIR / "ratings.csv", OUT_RATINGS)

    # Get paper IDs from old ratings
    with open(OLD_DIR / "ratings.csv") as f:
        rows = list(csv.DictReader(f))
    paper_ids = [r["paper_id"] for r in rows]
    print(f"{len(paper_ids)} papers to process")

    # Download PDFs
    print("Authenticating with OpenReview...")
    client = get_or_client()

    success = 0
    failed = 0
    for i, pid in enumerate(paper_ids, 1):
        md_path = OUT_PAPERS / f"{pid}.txt"
        if md_path.exists() and md_path.stat().st_size > 100:
            success += 1
            continue

        pdf_path = OUT_PDFS / f"{pid}.pdf"
        if not download_pdf(client, pid, pdf_path):
            print(f"  [{i}/{len(paper_ids)}] SKIPPED (download): {pid}")
            failed += 1
            continue
        time.sleep(0.5)

        text = pdf_to_markdown(pdf_path)
        if not text or len(text) < 500:
            print(f"  [{i}/{len(paper_ids)}] SKIPPED (conversion): {pid}")
            failed += 1
            continue

        md_path.write_text(text, encoding="utf-8")
        success += 1
        if i % 20 == 0:
            print(f"  {i}/{len(paper_ids)} done...")

    print(f"\nDone! {success} converted, {failed} failed.")


if __name__ == "__main__":
    main()
