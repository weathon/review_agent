from __future__ import annotations

"""
Fetch all ICLR papers with official reviews and export each paper to markdown.

Each markdown file contains:
- paper title
- abstract
- final decision
- all official human reviews, including scores

Usage:
  python fetch_iclr_reviews.py
  python fetch_iclr_reviews.py --year 2025
  python fetch_iclr_reviews.py --output-dir human_reviews
"""

import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import openreview
from dotenv import load_dotenv

load_dotenv()


def main():
    year = 2025
    output_dir = Path(__file__).parent / "iclr2025_abstract_reviews"
    existing_papers_dir = Path(__file__).parent / "iclr2025" / "papers"
    balanced = "--balanced" in sys.argv
    seed = 42
    n_samples = None

    if "--year" in sys.argv:
        year_index = sys.argv.index("--year")
        if year_index + 1 >= len(sys.argv):
            raise RuntimeError("--year requires a value")
        year = int(sys.argv[year_index + 1])
        output_dir = Path(__file__).parent / f"iclr{year}_abstract_reviews"

    if "--output-dir" in sys.argv:
        output_index = sys.argv.index("--output-dir")
        if output_index + 1 >= len(sys.argv):
            raise RuntimeError("--output-dir requires a value")
        output_dir = Path(sys.argv[output_index + 1])

    if "--existing-papers-dir" in sys.argv:
        existing_index = sys.argv.index("--existing-papers-dir")
        if existing_index + 1 >= len(sys.argv):
            raise RuntimeError("--existing-papers-dir requires a value")
        existing_papers_dir = Path(sys.argv[existing_index + 1])

    args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--year", "--output-dir", "--existing-papers-dir"}:
            skip_next = True
            continue
        if arg.startswith("--"):
            continue
        args.append(arg)
    if len(args) > 0:
        n_samples = int(args[0])
    if len(args) > 1:
        seed = int(args[1])

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in .env\n"
            "Sign up at https://openreview.net/signup"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_paper_ids = set()
    if existing_papers_dir.exists():
        for path in existing_papers_dir.glob("*"):
            if path.is_file():
                existing_paper_ids.add(path.stem)
        print(f"Loaded {len(existing_paper_ids)} existing paper ids from {existing_papers_dir}")
    else:
        print(f"Existing papers dir does not exist, skipping exclusion: {existing_papers_dir}")

    client = openreview.api.OpenReviewClient(
        username=username,
        password=password,
        baseurl="https://api2.openreview.net",
    )

    venue = f"ICLR.cc/{year}/Conference"
    print(f"Fetching submissions for {venue}...")
    venue_group = client.get_group(venue)
    submission_name = venue_group.content["submission_name"]["value"]
    notes = client.get_all_notes(
        invitation=f"{venue}/-/{submission_name}",
        details="directReplies",
    )
    print(f"Fetched {len(notes)} submission notes.")

    papers = []
    skipped_count = 0
    skipped_existing_count = 0

    for note in notes:
        try:
            content = note.content
            details = note.details

            venue_value = content.get("venue", "")
            if isinstance(venue_value, dict):
                venue_value = venue_value.get("value", "")

            if "Desk" in venue_value:
                skipped_count += 1
                continue

            title = content.get("title", "")
            if isinstance(title, dict):
                title = title.get("value", "")

            abstract = content.get("abstract", "")
            if isinstance(abstract, dict):
                abstract = abstract.get("value", "")

            reviews = []
            decision = None

            for reply in details.get("directReplies", []):
                invitations = reply.get("invitations", [])
                review_content = reply.get("content", {})

                if any(inv.endswith("/-/Decision") for inv in invitations):
                    decision = review_content.get("decision", "")
                    if isinstance(decision, dict):
                        decision = decision.get("value", "")

                if not any(inv.endswith("/-/Official_Review") for inv in invitations):
                    continue

                review = {}
                for field in [
                    "summary",
                    "strengths",
                    "weaknesses",
                    "questions",
                    "limitations",
                    "soundness",
                    "presentation",
                    "contribution",
                    "rating",
                    "confidence",
                ]:
                    if field not in review_content:
                        continue
                    value = review_content[field]
                    if isinstance(value, dict):
                        value = value.get("value", "")
                    if value in ("", None):
                        continue
                    review[field] = value

                rating_value = review.get("rating")
                if isinstance(rating_value, str):
                    rating_prefix = rating_value.split(":", 1)[0].strip()
                    if rating_prefix.isdigit():
                        review["rating_number"] = int(rating_prefix)
                elif isinstance(rating_value, (int, float)):
                    review["rating_number"] = int(rating_value)

                reviews.append(review)

            if not reviews:
                skipped_count += 1
                continue
            numeric_scores = [review["rating_number"] for review in reviews if "rating_number" in review]

            if "Withdrawn" in venue_value and not decision and len(numeric_scores) > 0:
                # raise RuntimeError(f"Missing decision for withdrawn paper {note.id}")
                decision = "Withdrawn (Treated as Reject)"
                print(f"Warning: Paper {note.id} is marked as Withdrawn but has no decision field, treating as Withdrawn.")
            if len(numeric_scores) < 0:
                raise RuntimeError(f"No numeric ratings found for paper {note.id}")
            # Withdrawn treated as reject
            if "Withdrawn" not in venue_value and not decision:
                raise RuntimeError(f"Missing decision for paper {note.id}")

            if note.id in existing_paper_ids:
                skipped_existing_count += 1
                continue

            if not numeric_scores:
                raise RuntimeError(f"Could not parse any numeric ratings for paper {note.id}")

            papers.append(
                {
                    "paper_id": note.id,
                    "title": title,
                    "abstract": abstract,
                    "decision": decision,
                    "reviews": reviews,
                    "avg_score": sum(numeric_scores) / len(numeric_scores),
                }
            )

        except Exception as exc:
            print(f"Skipping {getattr(note, 'id', 'unknown')}: {exc}")
            skipped_count += 1

    if balanced and n_samples is None:
        raise RuntimeError("Balanced mode requires a sample size, example: python3 fetch_iclr_reviews.py 100 42 --balanced")

    if n_samples is not None and n_samples > len(papers):
        print(f"Requested {n_samples} papers but only {len(papers)} are available after filtering.")
        n_samples = len(papers)

    selected_papers = papers
    if balanced:
        rng = random.Random(seed)
        bins = defaultdict(list)
        for paper in papers:
            bins[round(paper["avg_score"])].append(paper)
        for score_bin in bins:
            rng.shuffle(bins[score_bin])
        sorted_bins = sorted(bins.keys())
        n_bins = len(sorted_bins)
        per_bin = n_samples // n_bins
        remainder = n_samples % n_bins
        print(f"Stratified: {n_bins} bins, {per_bin}/bin (+{remainder} extra)")
        for score_bin in sorted_bins:
            print(f"  Score ~{score_bin}: {len(bins[score_bin])} papers")
        selected_papers = []
        for index, score_bin in enumerate(sorted_bins):
            take = min(per_bin + (1 if index < remainder else 0), len(bins[score_bin]))
            selected_papers.extend(bins[score_bin][:take])
        rng.shuffle(selected_papers)
        print(f"Sampled: {len(selected_papers)}\n")
    elif n_samples is not None:
        random.seed(seed)
        selected_papers = random.sample(papers, n_samples)
        print(f"Random sampled: {len(selected_papers)} papers with seed {seed}\n")
    else:
        print(f"Exporting all {len(selected_papers)} eligible papers\n")

    saved_count = 0
    for paper in selected_papers:
        markdown_lines = [
            f"# {paper['title']}",
            "",
            f"- Decision: {paper['decision']}",
            f"- Scores: {', '.join(str(r.get('rating_number', r.get('rating', 'N/A'))) for r in paper['reviews'])}",
            "",
            "## Abstract",
            paper["abstract"],
            "",
            "## Human Reviews",
            "",
        ]

        review_sections = []
        for index, review in enumerate(paper["reviews"], start=1):
            section_lines = [f"## Human Reviewer {index}"]

            if "rating" in review:
                section_lines.append(f"### Rating\n{review['rating']}")
            if "rating_number" in review:
                section_lines.append(f"### Rating Number\n{review['rating_number']}")
            if "confidence" in review:
                section_lines.append(f"### Confidence\n{review['confidence']}")

            for field in [
                "summary",
                "strengths",
                "weaknesses",
                "questions",
                "limitations",
                "soundness",
                "presentation",
                "contribution",
            ]:
                if field in review:
                    section_lines.append(
                        f"### {field.replace('_', ' ').title()}\n{review[field]}"
                    )

            review_sections.append("\n\n".join(section_lines))

        markdown_lines.append("\n\n---\n\n".join(review_sections))
        markdown_text = "\n".join(markdown_lines).strip() + "\n"

        filename = re.sub(r"[^\w.-]+", "_", paper["paper_id"].strip()).strip("._")
        if not filename:
            raise RuntimeError(f"Could not build filename from '{paper['paper_id']}'")

        output_path = output_dir / f"{filename}.md"
        output_path.write_text(markdown_text, encoding="utf-8")
        saved_count += 1

        if saved_count % 50 == 0:
            print(f"Saved {saved_count} markdown files so far...")

    print(f"Done. Saved {saved_count} papers to {output_dir}")
    print(f"Skipped {skipped_count} notes.")
    print(f"Skipped {skipped_existing_count} papers already present in {existing_papers_dir}.")


if __name__ == "__main__":
    main()
