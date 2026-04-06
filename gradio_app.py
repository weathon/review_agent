from __future__ import annotations

import asyncio
import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Generator

import gradio as gr

from fetch_iclr2026 import pdf_to_markdown
from paper_reviewer import review_paper_text


DEFAULT_CALIBRATION_PATH = Path("calibration.md")
DEFAULT_BENCHMARK_PATH = Path("bench_scores.csv")
APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.stone,
)
APP_CSS = """
.gradio-container { max-width: 1120px !important; }
#run-btn {
  background: linear-gradient(135deg, #0f766e, #0f766e 35%, #155e75);
  border: none !important;
}
#run-btn:hover {
  filter: brightness(1.06);
}
"""


def _load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _load_paper_input(path: str) -> tuple[str, str]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        text = pdf_to_markdown(input_path)
        if not text.strip():
            raise gr.Error("PDF parsing failed. No usable text was extracted.")
        return text, input_path.with_suffix(".txt").name
    return _load_text_file(path), input_path.name


def _extract_section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    tail = text[start:]
    if end_marker:
        end = tail.find(end_marker)
        if end != -1:
            tail = tail[:end]
    return tail.strip()


@lru_cache(maxsize=1)
def _load_benchmark_stats() -> dict[str, float | int | None]:
    if not DEFAULT_BENCHMARK_PATH.exists():
        return {}

    rows: list[tuple[float, int]] = []
    with DEFAULT_BENCHMARK_PATH.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_score = (row.get("pred_score") or "").strip()
            gt_binary = (row.get("gt_binary") or "").strip().lower()
            if not pred_score or gt_binary not in {"accept", "reject"}:
                continue
            try:
                rows.append((float(pred_score), 1 if gt_binary == "accept" else 0))
            except ValueError:
                continue

    if not rows:
        return {}

    scores = [score for score, _ in rows]
    labels = [label for _, label in rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"n": len(rows), "auroc": None, "threshold": None}

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(ranked):
        end = idx + 1
        while end < len(ranked) and ranked[end][1] == ranked[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for original_idx, _ in ranked[idx:end]:
            ranks[original_idx] = avg_rank
        idx = end
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    best_threshold = None
    best_j = float("-inf")
    best_tpr = 0.0
    best_fpr = 0.0
    for threshold in sorted(set(scores), reverse=True):
        tp = fp = tn = fn = 0
        for score, label in rows:
            predicted_accept = score >= threshold
            if predicted_accept and label == 1:
                tp += 1
            elif predicted_accept and label == 0:
                fp += 1
            elif label == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        j_score = tpr - fpr
        if j_score > best_j:
            best_j = j_score
            best_threshold = threshold
            best_tpr = tpr
            best_fpr = fpr

    predicted_accept_rows = [(score, label) for score, label in rows if score >= best_threshold]
    predicted_reject_rows = [(score, label) for score, label in rows if score < best_threshold]
    accept_side_accepts = sum(label for _, label in predicted_accept_rows)
    reject_side_accepts = sum(label for _, label in predicted_reject_rows)

    return {
        "n": len(rows),
        "auroc": auroc,
        "threshold": best_threshold,
        "tpr": best_tpr,
        "fpr": best_fpr,
        "accept_side_total": len(predicted_accept_rows),
        "accept_side_accepts": accept_side_accepts,
        "reject_side_total": len(predicted_reject_rows),
        "reject_side_accepts": reject_side_accepts,
        "scores": tuple(scores),
    }


def _score_percentile(score: float) -> str:
    stats = _load_benchmark_stats()
    scores = stats.get("scores")
    if not scores:
        return ""

    benchmark_scores = [float(value) for value in scores]
    below = sum(1 for value in benchmark_scores if value < score)
    equal = sum(1 for value in benchmark_scores if value == score)
    percentile = 100.0 * (below + 0.5 * equal) / len(benchmark_scores)
    return (
        f"- Percentile vs benchmark predicted scores: **{percentile:.1f}th** "
        f"({below} below, {equal} equal, midpoint-rank over {len(benchmark_scores)} benchmark scores)"
    )


def _acceptance_summary(score: float) -> str:
    stats = _load_benchmark_stats()
    threshold = stats.get("threshold")
    auroc = stats.get("auroc")
    if threshold is None or auroc is None:
        return ""

    on_accept_side = score >= float(threshold)
    side_prefix = "accept" if on_accept_side else "reject"
    side_total = int(stats[f"{side_prefix}_side_total"])
    side_accepts = int(stats[f"{side_prefix}_side_accepts"])
    side_rate = (side_accepts / side_total) if side_total else 0.0
    predicted_label = "Accept" if on_accept_side else "Reject"
    distance = abs(score - float(threshold))

    lines = [
        "## Acceptance Likelihood",
        "",
        f"- Predicted label at benchmark threshold: **{predicted_label}**",
        f"- Score threshold from benchmark ROC sweep: **{float(threshold):.2f}**",
        f"- Empirical accept rate on this side of the threshold: **{side_accepts}/{side_total} = {side_rate:.0%}**",
        f"- Benchmark AUROC for score -> accept/reject: **{float(auroc):.3f}**",
        f"- Distance from threshold: **{distance:.2f}**",
        "",
        f"*Derived from `{DEFAULT_BENCHMARK_PATH.name}` ({int(stats['n'])} benchmark papers), not hardcoded score buckets.*",
    ]
    return "\n".join(lines)


def _format_final_review(full_output: str) -> str:
    # Extract the review markdown between the FINAL CONSOLIDATED REVIEW and PREDICTED SCORE headers
    match = re.search(
        r"FINAL CONSOLIDATED REVIEW.*?\n=+\n\n([\s\S]*?)\n=+\nPREDICTED SCORE",
        full_output,
    )
    review_text = match.group(1).strip() if match else ""
    if not review_text:
        return "Could not extract the final review from the output."

    score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", full_output)
    decision_match = re.search(r"Decision:\s*(Accept|Reject|N/A)", full_output)

    parts = []
    if score_match or decision_match:
        score_lines = ["## Score", ""]
        if score_match:
            score_value = float(score_match.group(1))
            score_lines.append(f"- Score: **{score_value:.1f}**")
            percentile_line = _score_percentile(score_value)
            if percentile_line:
                score_lines.append(percentile_line)
        if decision_match:
            score_lines.append(f"- Decision: **{decision_match.group(1)}**")
        parts.append("\n".join(score_lines))

    if score_match:
        acceptance_summary = _acceptance_summary(float(score_match.group(1)))
        if acceptance_summary:
            parts.append(acceptance_summary)

    parts.append(review_text)

    return "\n\n".join(part for part in parts if part.strip())


def run_review(
    api_key: str,
    uploaded_file,
    pasted_text: str,
    venue: str,
    parallel: bool,
    use_related_work: bool,
    use_spark: bool,
    use_calibration: bool,
) -> Generator[tuple[str, str, str | None], None, None]:
    api_key = (api_key or "").strip()
    if not api_key:
        raise gr.Error("Please enter an OpenRouter API key.")

    yield "Running: reading input...", "Review started. Preparing input.", None

    paper_text = ""
    source_name = "paper.txt"

    if uploaded_file is not None:
        paper_text, source_name = _load_paper_input(uploaded_file)
    elif pasted_text and pasted_text.strip():
        paper_text = pasted_text
    else:
        raise gr.Error("Please upload a `.pdf` / `.txt` / `.md` file, or paste the paper text directly.")

    calibration_path = None
    if use_calibration and DEFAULT_CALIBRATION_PATH.exists():
        calibration_path = str(DEFAULT_CALIBRATION_PATH)

    yield (
        f"Running: loaded `{source_name}`, calling the reviewer...",
        "Review in progress. This may take from tens of seconds to a few minutes depending on paper length and enabled agents.",
        None,
    )

    review_output, saved_path = asyncio.run(
        review_paper_text(
            paper_text=paper_text,
            source_name=source_name,
            parallel=parallel,
            skip_related_work=not use_related_work,
            skip_spark=not use_spark,
            venue=(venue or "").strip(),
            calibration_path=calibration_path,
            api_key=api_key,
            output_dir="webui_runs",
        )
    )

    summary = f"Done. Full output saved to `{saved_path}`."
    yield summary, _format_final_review(review_output), saved_path


with gr.Blocks(title="Paper Reviewer", theme=APP_THEME, css=APP_CSS) as demo:
    gr.Markdown(
        """
        # Multi-Agent Paper Reviewer
        Local-only, no auth, BYOK. Enter an OpenRouter API key, then upload a paper file or paste the paper text directly.
        PDF upload is supported and reuses the parser from the dataset builder.
        """
    )

    with gr.Row():
        api_key = gr.Textbox(
            label="OpenRouter API key",
            type="password",
            placeholder="sk-...",
        )
        venue = gr.Textbox(
            label="Venue",
            placeholder="ICLR / NeurIPS / ICML",
            value="ICLR",
        )

    with gr.Row():
        uploaded_file = gr.File(
            label="Paper File",
            file_types=[".pdf", ".txt", ".md"],
            type="filepath",
        )
        pasted_text = gr.Textbox(
            label="Or Paste Paper Text",
            lines=18,
            placeholder="Paste the title, abstract, and main text here",
        )

    with gr.Row():
        parallel = gr.Checkbox(label="Run Agents In Parallel", value=True)
        use_related_work = gr.Checkbox(label="Enable Related Work", value=False)
        use_spark = gr.Checkbox(label="Enable Spark Finder", value=True)
        use_calibration = gr.Checkbox(
            label="Use calibration.md",
            value=DEFAULT_CALIBRATION_PATH.exists(),
        )

    run_button = gr.Button("Start Review", variant="primary", elem_id="run-btn")
    status = gr.Markdown()
    output = gr.Markdown(label="Final Review")
    download_file = gr.File(label="Download Full Output", visible=True)

    run_button.click(
        fn=run_review,
        inputs=[
            api_key,
            uploaded_file,
            pasted_text,
            venue,
            parallel,
            use_related_work,
            use_spark,
            use_calibration,
        ],
        outputs=[status, output, download_file],
        show_progress="full",
    )

demo.queue()


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
