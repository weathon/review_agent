"""
One-off script: rename cal/*.md → cal/*_review.md and copy paper texts to cal/*_paper.md.
"""
import csv
import json
import re
from pathlib import Path


def shorten_title(title: str, max_len: int = 60) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", title.lower())
    name = re.sub(r"\s+", "_", name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "untitled"


cal_dir = Path("cal")
papers_dir = Path("iclr2026_balanced/papers")
ratings_path = Path("iclr2026_balanced/ratings.csv")
cal_ids_path = Path("calibration_ids.json")

# Load calibration IDs
cal_ids = set(json.loads(cal_ids_path.read_text()))

# Build paper_id → title mapping from ratings
with open(ratings_path) as f:
    reader = csv.DictReader(f)
    id_to_title = {r["paper_id"]: r["title"] for r in reader}

# Build short_name → paper_id mapping for calibration papers
name_to_pid = {}
for pid in cal_ids:
    title = id_to_title.get(pid, pid)
    short = shorten_title(title)
    name_to_pid[short] = pid

# Rename existing .md files → _review.md and copy paper text
renamed = 0
copied = 0
for md_file in sorted(cal_dir.glob("*.md")):
    stem = md_file.stem
    # Skip if already migrated
    if stem.endswith("_review") or stem.endswith("_paper"):
        continue

    # Rename to _review.md
    new_review = cal_dir / f"{stem}_review.md"
    md_file.rename(new_review)
    print(f"  Renamed: {md_file.name} → {new_review.name}")
    renamed += 1

    # Copy paper text
    pid = name_to_pid.get(stem)
    if pid:
        paper_src = papers_dir / f"{pid}.txt"
        if paper_src.exists():
            paper_text = paper_src.read_text(encoding="utf-8", errors="replace")
            paper_dst = cal_dir / f"{stem}_paper.md"
            paper_dst.write_text(paper_text, encoding="utf-8")
            print(f"  Copied:  {paper_src.name} → {paper_dst.name}")
            copied += 1
        else:
            print(f"  WARNING: paper text not found: {paper_src}")
    else:
        print(f"  WARNING: no paper_id mapping for: {stem}")

print(f"\nDone: {renamed} renamed, {copied} paper texts copied")
