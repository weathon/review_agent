"""
LLM-as-Judge: compare two AI-generated reviews against human reviews.

Fetches human reviews from OpenReview on demand, then uses Claude Agent SDK
to judge which of two candidate reviews is better aligned with the human review.

Usage:
  python llm_as_judge.py <paper_id> <review1_path> <review2_path> [--dataset unbalanced|balanced]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATASETS = {
    "unbalanced": Path(__file__).parent / "iclr2026_unbalanced",
    "balanced": Path(__file__).parent / "iclr2026_balanced",
}


def fetch_human_reviews(paper_id: str) -> str:
    """Fetch human reviews from OpenReview API and return as formatted text."""
    import openreview

    client = openreview.api.OpenReviewClient(
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ["OPENREVIEW_PASSWORD"],
        baseurl="https://api2.openreview.net",
    )

    replies = client.get_all_notes(forum=paper_id)
    reviews = [
        r for r in replies
        if any("Official_Review" in inv for inv in r.invitations)
    ]

    if not reviews:
        raise ValueError(f"No human reviews found for paper {paper_id}")

    sections = []
    for i, rev in enumerate(reviews, 1):
        rc = rev.content
        parts = [f"## Human Reviewer {i}"]
        for field in ["summary", "strengths", "weaknesses", "rating", "confidence"]:
            if field in rc:
                val = rc[field].get("value", "") if isinstance(rc[field], dict) else rc[field]
                parts.append(f"### {field.title()}\n{val}")
        sections.append("\n\n".join(parts))

    return "\n\n---\n\n".join(sections)


def find_paper_info(paper_id: str, dataset: str) -> dict | None:
    """Look up paper metadata from the dataset's all_notes.json."""
    data_dir = DATASETS.get(dataset)
    if not data_dir:
        return None
    notes_file = data_dir / "all_notes.json"
    if not notes_file.exists():
        return None
    with open(notes_file) as f:
        all_notes = json.load(f)
    for note in all_notes:
        if note["paper_id"] == paper_id:
            return note
    return None


async def judge(paper_id: str, review1_path: str, review2_path: str, dataset: str = "unbalanced") -> str:
    """Run the LLM judge comparing two reviews against human reviews."""
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    # Fetch human reviews
    print(f"Fetching human reviews for {paper_id} from OpenReview...")
    human_reviews_text = fetch_human_reviews(paper_id)

    # Write human reviews to a temp file so the agent can Read it
    tmp_dir = Path(tempfile.mkdtemp(prefix="judge_"))
    human_path = tmp_dir / "human_reviews.txt"
    human_path.write_text(human_reviews_text, encoding="utf-8")

    # Look up paper info for context
    paper_info = find_paper_info(paper_id, dataset)
    context = ""
    if paper_info:
        context = f"\nPaper title: {paper_info['title']}\n"

    review1_abs = str(Path(review1_path).resolve())
    review2_abs = str(Path(review2_path).resolve())
    human_abs = str(human_path.resolve())

    prompt = f"""\
You are an expert meta-reviewer judging the quality of AI-generated paper reviews.

You have three files to read:
1. Expert human reviews (ground truth): {human_abs}
2. Review A (candidate): {review1_abs}
3. Review B (candidate): {review2_abs}
{context}
Your task:
1. Read all three files carefully.
2. For each candidate review (A and B), analyze the following dimensions by comparing against the expert reviews:
For each review, evaluate: Does the review has same reasonability and preference as the human, even though their points are different.

Do NOT treat long and "through" reviews as good, check the actual content. 

At the very end of your response, output exactly one of these lines:
**Final Decision**: **Review A** aligns better with the expert reviews.
**Final Decision**: **Review B** aligns better with the expert reviews.
**Final Decision**: **Tie** - both reviews align equally with the expert reviews.
"""

    print("Running Claude judge agent...")
    result_text = ""
    options = ClaudeAgentOptions(
        model="claude-opus-4-6",
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        effort="high",
        max_turns=10,
    )
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text

    # Clean up
    human_path.unlink(missing_ok=True)
    tmp_dir.rmdir()

    # Extract winner from "**Final Decision**: **Review A/B** ..." or "**Tie**"
    winner = "Unknown"
    for line in result_text.strip().splitlines()[::-1]:
        line = line.strip()
        if "Final Decision" in line:
            if "Review A" in line and "Review B" not in line.split("Review A")[1].split("aligns")[0]:
                winner = "Review A"
            elif "Review B" in line and "Review A" not in line.split("Review B")[1].split("aligns")[0]:
                winner = "Review B"
            elif "Tie" in line:
                winner = "Tie"
            break

    print(f"\n{'=' * 72}")
    print(result_text)
    print(f"\n{'=' * 72}")
    print(f"WINNER: {winner}")
    return winner


def main():
    if len(sys.argv) < 4:
        print("Usage: python  llm_as_judge.py <paper_id> <review1_path> <review2_path> [--dataset unbalanced|balanced]")
        sys.exit(1)

    paper_id = sys.argv[1]
    review1_path = sys.argv[2]
    review2_path = sys.argv[3]

    dataset = "unbalanced"
    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            dataset = sys.argv[idx + 1]

    # Check dataset exists
    for ds_name, ds_path in DATASETS.items():
        notes = ds_path / "all_notes.json"
        if notes.exists():
            info = find_paper_info(paper_id, ds_name)
            if info:
                dataset = ds_name
                print(f"Found paper in {ds_name} dataset")
                break

    asyncio.run(judge(paper_id, review1_path, review2_path, dataset))


if __name__ == "__main__":
    main()
