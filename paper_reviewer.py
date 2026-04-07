from __future__ import annotations

"""
Thin wrapper around review_paper.sh for Python callers (gradio, bench scripts).
Each agent is a standalone script in agents/. Orchestration is in review_paper.sh.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = Path(__file__).parent
REVIEW_SCRIPT = SCRIPT_DIR / "review_paper.sh"

# Config — importable by bench scripts
base_model = "deepseek/deepseek-v3.2"
MODEL_HARSH = "claude:claude-opus-4-6"
MODEL_NEUTRAL = f"{base_model}"
MODEL_SPARK = f"{base_model}"
MODEL_RELATED_WORK = f"{base_model}:online"
MODEL_MERGER = f"{base_model}"


def sanitize_text(text: str) -> str:
    return text.replace("\x00", "")


def score_to_decision(score: float | None) -> str | None:
    return "N/A"


def decision_match(predicted: str | None, gt_binary: str) -> bool | None:
    if predicted in (None, "", "N/A"):
        return None
    return predicted == gt_binary


def match_label(match: bool | None) -> str:
    if match is None:
        return "N/A"
    return "YES" if match else "NO"


def review_paper(
    paper_path: str,
    parallel: bool = True,
    skip_related_work: bool = True,
    skip_spark: bool = False,
    skip_neutral: bool = False,
    venue: str = "ICLR",
    calibration_context: str = "",
    cal_dir: str = "",
    calibration_path: str | None = None,
    api_key: str | None = None,
) -> tuple[str, float]:
    """Run review_paper.sh. Returns (review_text, 0.0)."""
    path = Path(paper_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Paper not found: {path}")

    work_dir = tempfile.mkdtemp(prefix="review_")
    cmd = ["bash", str(REVIEW_SCRIPT), str(path), work_dir]

    env = os.environ.copy()
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key

    print(f"Running: bash review_paper.sh {path} {work_dir}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"review_paper.sh failed (exit {result.returncode})")

    # Read outputs
    merged = Path(work_dir) / "merged.txt"
    score_file = Path(work_dir) / "score.txt"
    review_text = merged.read_text(encoding="utf-8") if merged.exists() else ""
    score = float(score_file.read_text().strip()) if score_file.exists() else 0.0

    # Write combined output file
    output_path = path.parent / f"{path.stem}_review.md"
    output_path.write_text(f"{review_text}\n\nScore: {score}\n", encoding="utf-8")

    return review_text, 0.0


def review_paper_text(
    paper_text: str,
    source_name: str = "paper.txt",
    parallel: bool = True,
    skip_related_work: bool = True,
    skip_spark: bool = False,
    skip_neutral: bool = False,
    venue: str = "ICLR",
    calibration_context: str = "",
    cal_dir: str = "",
    calibration_path: str | None = None,
    api_key: str | None = None,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Review paper content provided directly as text."""
    cleaned_text = sanitize_text(paper_text)
    if not cleaned_text.strip():
        raise ValueError("Paper content is empty.")

    target_dir = Path(output_dir or "webui_runs").expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(source_name).name or "paper.txt"
    if not Path(safe_name).suffix:
        safe_name = f"{safe_name}.txt"
    input_path = target_dir / safe_name
    input_path.write_text(cleaned_text, encoding="utf-8")

    result, _ = review_paper(str(input_path), parallel=parallel, skip_related_work=skip_related_work,
                             skip_spark=skip_spark, skip_neutral=skip_neutral, venue=venue,
                             calibration_context=calibration_context, cal_dir=cal_dir,
                             calibration_path=calibration_path, api_key=api_key)
    return result, str(input_path.with_name(f"{input_path.stem}_review.md"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python paper_reviewer.py <paper.txt>")
        sys.exit(1)
    cmd = ["bash", str(REVIEW_SCRIPT)] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
