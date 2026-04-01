from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Generator

import gradio as gr

from fetch_iclr2025 import pdf_to_markdown
from paper_reviewer import review_paper_text


DEFAULT_CALIBRATION_PATH = Path("calibration.md")
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
            score_lines.append(f"- Score: **{score_match.group(1)}**")
        if decision_match:
            score_lines.append(f"- Decision: **{decision_match.group(1)}**")
        parts.append("\n".join(score_lines))

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

    calibration_context = ""
    if use_calibration and DEFAULT_CALIBRATION_PATH.exists():
        calibration_context = DEFAULT_CALIBRATION_PATH.read_text(encoding="utf-8", errors="replace")

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
            calibration_context=calibration_context,
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
        use_related_work = gr.Checkbox(label="Enable Related Work", value=True)
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
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
