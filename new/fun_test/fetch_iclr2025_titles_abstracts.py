import json
import os
from pathlib import Path

from dotenv import load_dotenv
import openreview

load_dotenv()

YEAR = 2025
OUT = Path(__file__).parent / "iclr2025_titles_abstracts.json"


def get_client():
    return openreview.api.OpenReviewClient(
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ["OPENREVIEW_PASSWORD"],
        baseurl="https://api2.openreview.net",
    )


def main():
    client = get_client()
    venue = f"ICLR.cc/{YEAR}/Conference"
    venue_group = client.get_group(venue)
    submission_name = venue_group.content["submission_name"]["value"]
    print(f"Fetching submissions for {venue}...")
    notes = client.get_all_notes(
        invitation=f"{venue}/-/{submission_name}", details="directReplies"
    )
    print(f"Got {len(notes)} submissions.")

    out = []
    for n in notes:
        c = n.content
        scores = []
        for reply in (n.details or {}).get("directReplies", []):
            if not any(inv.endswith("/-/Official_Review") for inv in reply.get("invitations", [])):
                continue
            rating_val = reply.get("content", {}).get("rating", {}).get("value", "")
            if isinstance(rating_val, str) and ":" in rating_val:
                try:
                    scores.append(int(rating_val.split(":")[0].strip()))
                except ValueError:
                    pass
            elif isinstance(rating_val, (int, float)):
                scores.append(int(rating_val))
        avg_score = sum(scores) / len(scores) if scores else None
        out.append({
            "paper_id": n.id,
            "title": c.get("title", {}).get("value", ""),
            "abstract": c.get("abstract", {}).get("value", ""),
            "scores": scores,
            "avg_score": avg_score,
        })

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out)} entries to {OUT}")


if __name__ == "__main__":
    main()
