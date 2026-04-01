"""
Fetch a sample of ICLR 2025 papers with full text and reviewer scores.

Downloads PDFs from OpenReview, converts to markdown, and saves a dataset
ready for benchmarking.

Requires:
  - OpenReview account (set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in .env)
  - pip install openreview-py pymupdf4llm polars tqdm

Usage:
  python fetch_iclr2025.py                  # 100 papers, seed=42
  python fetch_iclr2025.py 50 4112          # 50 papers, seed=4112
  python fetch_iclr2025.py 100 42 --balanced
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "iclr2025_data"
PAPERS_DIR = DATA_DIR / "papers"
PDFS_DIR = DATA_DIR / "pdfs"
RATINGS_FILE = DATA_DIR / "ratings.csv"
OPENREVIEW_URL = "https://openreview.net"


def fetch_notes():
    """Fetch all ICLR 2025 submission notes with reviews via OpenReview API."""
    client = get_or_client()

    venue = "ICLR.cc/2025/Conference"
    print(f"Fetching submissions for {venue}...")
    venue_group = client.get_group(venue)
    submission_name = venue_group.content["submission_name"]["value"]

    notes = client.get_all_notes(
        invitation=f"{venue}/-/{submission_name}", details="directReplies"
    )
    print(f"Got {len(notes)} submissions.")
    return notes


def parse_note(note) -> dict | None:
    """Parse a note into paper_id, title, abstract, pdf_url, scores, decision."""
    content = note.content
    details = note.details

    title = content.get("title", {}).get("value", "")
    abstract = content.get("abstract", {}).get("value", "")
    pdf_url = content.get("pdf", {}).get("value", "")
    venue = content.get("venue", {}).get("value", "")
    venueid = content.get("venueid", {}).get("value", "")

    # Skip desk rejected (no reviews), but keep withdrawn (may have reviews + low scores)
    if "Desk" in venue:
        return None

    # Parse reviews from direct replies
    scores = []
    direct_replies = details.get("directReplies", [])
    for reply in direct_replies:
        inv = reply.get("invitations", [""])[0]
        if "Official_Review" not in inv:
            continue
        rc = reply.get("content", {})
        rating_val = rc.get("rating", {}).get("value", "")
        # Rating is like "8: accept, good paper" — extract the number
        if isinstance(rating_val, str) and ":" in rating_val:
            try:
                scores.append(int(rating_val.split(":")[0].strip()))
            except ValueError:
                pass
        elif isinstance(rating_val, (int, float)):
            scores.append(int(rating_val))

    if not scores:
        return None

    # Parse decision
    decision = None
    for reply in direct_replies:
        inv = reply.get("invitations", [""])[0]
        if "Decision" in inv:
            rc = reply.get("content", {})
            decision = rc.get("decision", {}).get("value", "")
            break

    avg_score = sum(scores) / len(scores)
    # Withdrawn papers are treated as Reject
    if "Withdrawn" in venue:
        gt_binary = "Reject"
        decision = decision or "Withdrawn (treated as Reject)"
    elif decision:
        gt_binary = "Accept" if "Accept" in decision else "Reject"
    else:
        gt_binary = "Accept" if avg_score >= 5.5 else "Reject"

    return {
        "paper_id": note.id,
        "title": title,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "venue": venue,
        "scores": scores,
        "avg_score": avg_score,
        "decision": decision or f"Inferred ({gt_binary})",
        "gt_binary": gt_binary,
    }


def get_or_client():
    """Get an authenticated OpenReview API client."""
    import openreview
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in .env\n"
            "Sign up at https://openreview.net/signup"
        )
    return openreview.api.OpenReviewClient(
        username=username, password=password,
        baseurl="https://api2.openreview.net",
    )


def download_pdf(or_client, paper_id: str) -> Path | None:
    """Download a PDF from OpenReview using the authenticated API."""
    outfile = PDFS_DIR / f"{paper_id}.pdf"
    if outfile.exists() and outfile.stat().st_size > 0:
        return outfile

    try:
        pdf_bytes = or_client.get_pdf(paper_id)
        if len(pdf_bytes) > 1000:
            outfile.write_bytes(pdf_bytes)
            return outfile
        else:
            print(f"    Download failed: got {len(pdf_bytes)} bytes")
            return None
    except Exception as e:
        print(f"    Download error: {e}")
        return None


def pdf_to_markdown(pdf_path: Path) -> str:
    """Convert PDF to markdown text, cleaning up line numbers and artifacts."""
    import re
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


def stratified_sample(papers, n, seed):
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])
    sorted_bins = sorted(bins.keys())
    n_bins = len(sorted_bins)
    per_bin = n // n_bins
    remainder = n % n_bins
    print(f"Stratified: {n_bins} bins, {per_bin}/bin (+{remainder} extra)")
    for k in sorted_bins:
        print(f"  Score ~{k}: {len(bins[k])} papers")
    samples = []
    for i, k in enumerate(sorted_bins):
        take = min(per_bin + (1 if i < remainder else 0), len(bins[k]))
        samples.extend(bins[k][:take])
    rng.shuffle(samples)
    print(f"Sampled: {len(samples)}\n")
    return samples


def main(n_samples: int = 100, seed: int = 42, balanced: bool = False):
    print("=" * 72)
    print("ICLR 2025 Dataset Builder")
    print("=" * 72)

    # Create dirs
    DATA_DIR.mkdir(exist_ok=True)
    PAPERS_DIR.mkdir(exist_ok=True)
    PDFS_DIR.mkdir(exist_ok=True)

    # Check for cached notes
    cache_file = DATA_DIR / "all_notes.json"
    if cache_file.exists():
        print(f"Loading cached notes from {cache_file}...")
        with open(cache_file) as f:
            all_papers = json.load(f)
        print(f"Loaded {len(all_papers)} papers from cache.")
    else:
        # Fetch from API
        notes = fetch_notes()
        print("Parsing notes...")
        all_papers = []
        for note in notes:
            try:
                parsed = parse_note(note)
                if parsed:
                    all_papers.append(parsed)
            except Exception as e:
                continue
        print(f"Parsed {len(all_papers)} papers with reviews.")

        # Cache
        with open(cache_file, "w") as f:
            json.dump(all_papers, f)
        print(f"Cached to {cache_file}")

    # Sample
    if balanced:
        samples = stratified_sample(all_papers, n_samples, seed)
    else:
        random.seed(seed)
        samples = random.sample(all_papers, min(n_samples, len(all_papers)))
    print(f"Selected {len(samples)} papers.\n")

    # Download PDFs and convert to markdown
    print("Authenticating with OpenReview for PDF downloads...")
    or_client = get_or_client()

    success = 0
    with open(RATINGS_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "title", "decision", "gt_binary", "avg_score",
                     "score_0", "score_1", "score_2", "score_3", "score_4", "score_5"])

    for i, paper in enumerate(samples, 1):
        pid = paper["paper_id"]
        title = paper["title"]
        print(f"[{i}/{len(samples)}] {title[:60]}...")
        print(f"  Scores: {paper['scores']} avg={paper['avg_score']:.1f} dec={paper['gt_binary']}")

        # Check if already converted
        md_path = PAPERS_DIR / f"{pid}.txt"
        if md_path.exists() and md_path.stat().st_size > 100:
            print(f"  Already converted, skipping download.")
        else:
            # Download PDF
            print(f"  Downloading PDF...")
            pdf_path = download_pdf(or_client, pid)
            if not pdf_path:
                print(f"  SKIPPED (download failed)")
                continue

            # Convert to markdown
            print(f"  Converting to markdown...")
            text = pdf_to_markdown(pdf_path)
            if not text or len(text) < 500:
                print(f"  SKIPPED (conversion failed, got {len(text)} chars)")
                continue

            md_path.write_text(text, encoding="utf-8")
            print(f"  Saved: {len(text):,} chars")

        # Write to ratings CSV
        with open(RATINGS_FILE, "a", newline="") as f:
            w = csv.writer(f)
            scores_padded = paper["scores"] + [""] * (6 - len(paper["scores"]))
            w.writerow([pid, title, paper["decision"], paper["gt_binary"],
                         f"{paper['avg_score']:.2f}", *scores_padded[:6]])

        success += 1
        time.sleep(1)  # Be nice to OpenReview

    print(f"\n{'=' * 72}")
    print(f"Done! {success}/{len(samples)} papers downloaded and converted.")
    print(f"Papers dir: {PAPERS_DIR}")
    print(f"Ratings:    {RATINGS_FILE}")
    print(f"\nTo run the benchmark:")
    print(f"  python run_iclr_bench.py 10 42 --parallel --data-dir iclr2025_data")


if __name__ == "__main__":
    balanced = "--balanced" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if len(args) > 0 else 100
    seed = int(args[1]) if len(args) > 1 else 42
    main(n_samples=n, seed=seed, balanced=balanced)
