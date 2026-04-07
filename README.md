
## Participate in Our Evaluation

We are looking for participants to help evaluate our review quality. If you are interested, please contact us at wg25r@student.ubc.ca.

# :pineapple:  Multi-Agent Paper Reviewer

An automated academic paper review system that uses multiple LLM agents with different roles to produce a consolidated review and score. Benchmarked against ICLR 2026 ground truth reviews.

## Architecture

```
Phase 1 (all parallel):
  ├── Harsh Critic ────────── claude-sonnet-4-6 (Claude Agent SDK, reads paper with tools)
  ├── Neutral Reviewer ────── qwen/qwen3.5-flash-02-23 (OpenRouter)
  ├── Spark Finder ────────── qwen/qwen3.5-plus-02-15 (OpenRouter)
  └── Related Work Scout ──── qwen/qwen3.5-flash-02-23:online → qwen/qwen3.5-flash-02-23 filter (OpenRouter)

Phase 2 (two steps):
  ├── Merger ──────────────── z-ai/glm-5 via OpenRouter (consolidated review, no scores)
  └── Scorer ──────────────── claude-sonnet-4-6 (Claude Agent SDK, RAG over cal/ directory)
      └── Score Parser ────── openai/gpt-5.4-nano via OpenRouter (structured output extraction)
```

**Agents:**

| Agent | Model | Role |
|-------|-------|------|
| Harsh Critic | `claude-sonnet-4-6` (Agent SDK) | Section-by-section rubric review — raises genuine concerns, not nitpicks |
| Neutral Reviewer | `qwen/qwen3.5-flash-02-23` | Balanced assessment with strengths, weaknesses, novelty |
| Spark Finder | `qwen/qwen3.5-plus-02-15` | Identifies top 3-5 missing experiments, deeper analysis, obvious next steps |
| Related Work Scout | `qwen/qwen3.5-flash-02-23:online` | Proposes potentially missed references using OpenRouter's online model variant |
| Related Work Filter | `qwen/qwen3.5-flash-02-23` | Removes already-cited and loosely related results |
| Merger | `z-ai/glm-5` | Synthesizes all inputs into a final structured review (no scores) |
| Scorer | `claude-sonnet-4-6` (Agent SDK) | RAG-based scoring: searches calibration files via Grep/Read tools, compares paper against calibration examples |
| Score Parser | `openai/gpt-5.4-nano` | Extracts numerical score from scorer output using structured output |

The harsh critic and scorer both run via the Claude Agent SDK (with Read/Glob/Grep tool access). All other stages go through OpenRouter chat completions.

## Review And Scoring Design

The merger acts as an area chair and applies several filters:

- **Nonsense filter**: Removes criticisms that are factually wrong or misunderstand the paper
- **Nitpick filter**: Drops formatting, style, and minor phrasing complaints
- **Scope check**: For every weakness, asks:
  1. Is this a *real* weakness or a *nice-to-have*?
  2. Could this omission be *intentional* (scope decision, space constraint)?
  3. Is this within the paper's stated scope?

Real weaknesses go in `"weaknesses"` and inform the final assessment. Nice-to-haves go in `"nice_to_haves"` and should not be treated like core flaws. The merger outputs a structured review with no scores, then a separate scorer agent assigns the score.

### RAG-Based Scoring

The scorer uses the Claude Agent SDK with tool access (Grep, Read, Glob, Agent) to dynamically search a calibration directory (`cal/`) for relevant examples:

1. The scorer agent reads the consolidated review and paper text
2. It searches `cal/` for calibration papers with similar topics, methodologies, or quality levels
3. It reads the `_review.md` files (which include human scores) for 5-7 relevant matches
4. It compares the paper against those calibration examples on novelty, soundness, empirical support, significance, and clarity
5. A sub-agent can be spawned to summarize the paper in parallel with calibration search
6. The numerical score is extracted by `gpt-5.4-nano` with structured output parsing

The scorer also includes leakage detection — it checks for suspicious phrases in its output that might indicate the agent recognized a calibration paper as the same paper being reviewed.

## Motivation

Prior academic review agents have several practical problems that this project tries to avoid.

CSPaper Review (CSPR) appears to rely on a forced-score style of review generation. It forces the AI to generate a review for every single score in the score range (from 1-10) and then merge them. In practice, this design can encourage overly picky, internally inconsistent, or weakly grounded criticism, and may lead to contradictions across different parts of the review. Its related-work stage also appears to have relatively low precision, introducing a substantial amount of noisy or only weakly relevant feedback. Such noise can depress scores artificially and reduce the practical usability of the system for authors.

CSPR's calibration approach also appears to depend heavily on semantic analysis of generated critiques. This is potentially problematic because the number of negative points raised in a review is not a reliable proxy for overall paper quality. A paper may elicit many minor comments without having serious flaws, while a substantially stronger paper may have only a few high-impact concerns. As a result, direct semantic aggregation of negative points can distort score calibration. Combined with the previous point, this problem is actually exacerbated: when a paper has no major issues, the forced-low score agent is compelled to scrape the bottom of the barrel to find "weaknesses" that justify a score of 1. This led to many amusing incidents where reviewers raised points such as "why did the authors not define cosine similarity," simply because there was nothing else that could plausibly bring the paper down to that level. Although a merger module exists, these superficial points can still slip through, filling the review with trivial criticisms. When semantic classification is then applied, the review appears to reflect a lower quality assessment than is warranted.

A broader issue in this literature is evaluation methodology. Many systems emphasize MAE or exact agreement with a human score, but these metrics can be misleading on imbalanced datasets. For example, on a non-balanced conference sample, an always-predict-6 baseline can already achieve deceptively strong performance. Without careful stratification, such evaluation protocols can overstate model quality. CSPR also appears to exhibit selection bias in paper collection, which further limits confidence in the reported results.

The Stanford review agent appears to encounter similar related-work precision issues. It also exhibits a failure mode in which the model may incorrectly treat papers, methods, or models outside its training timeline as fabricated and challenge the user on that basis. Its reported correlation results are also difficult to interpret, since the presentation partly relies on human-human agreement rather than a cleaner AI-to-human benchmark. More broadly, like CSPR, it is neither open source nor accompanied by a sufficiently detailed technical description, which makes the system difficult to inspect, explain, or validate.

The Stanford system also predicts sub-dimensional scores (novelty, soundness, clarity, etc.) and then uses linear regression to combine them into a final score. This design is fragile because LLMs tend to produce inflated, non-discriminative subscores due to sycophancy — without external anchor points or calibration, the subscores cluster in a narrow range (e.g., 6-8 out of 10) regardless of actual paper quality. When the input features to the regression are themselves uninformative, the final predicted score inherits that lack of discrimination. In contrast, our system avoids subscores entirely: the merger produces a qualitative review with no numerical ratings, and the final score is assigned in a separate step anchored by RAG calibration with real human scores. This forces the scoring to be grounded in comparative quality rather than inflated self-assessments.

## Quick Start

```bash
# Clone
git clone https://github.com/weathon/review_agent.git
cd review_agent

# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter API key
echo 'OPENROUTER_API_KEY="sk-or-..."' > .env

# For fetching ICLR 2026 papers, also add OpenReview credentials:
echo 'OPENREVIEW_USERNAME="your@email.com"' >> .env
echo 'OPENREVIEW_PASSWORD="yourpassword"' >> .env
```

Additional dependencies used but not in `requirements.txt`: `pymupdf4llm`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `pydantic`, `claude-agent-sdk`.

## Usage

### Review a single paper

```bash
# Parallel mode is the default
python paper_reviewer.py paper.txt --venue NeurIPS

# Run sequentially instead
python paper_reviewer.py paper.txt --sequential --venue NeurIPS

# Enable related work search (disabled by default)
python paper_reviewer.py paper.txt --with-related-work

# Skip specific agents
python paper_reviewer.py paper.txt --no-spark --no-neutral
```

### Run the local Web UI

```bash
./run_webui.sh
```

Then open `http://127.0.0.1:7860`.

The UI features:
- no auth
- BYOK via an OpenRouter API key field in the page
- upload a `.pdf` / `.txt` / `.md` paper file, or paste paper text directly
- PDF parsing reuses the parser from `fetch_iclr2026.py`
- toggles for parallel mode, related work, spark finder, and calibration
- score percentile display relative to benchmark scores
- acceptance likelihood derived from benchmark AUROC/threshold

### Run everything (fetch → calibrate → benchmark)

```bash
./run_all.sh
```

This runs the benchmark pipeline:
1. Prints dataset distribution from `iclr2026_balanced/`
2. Runs reviewer benchmark with calibration and RAG scoring (200 papers, 3 concurrent)
3. Computes metrics + generates plots

### Step by step

```bash
# 1. Fetch ICLR 2026 papers (balanced for calibration, unbalanced for testing)
python fetch_iclr2026.py 100 42 --balanced --data-dir iclr2026_balanced
python fetch_iclr2026.py 100 42 --data-dir iclr2026_unbalanced

# 2. Build calibration set from balanced data (saves to cal/ as individual file pairs)
python build_calibration.py --data-dir iclr2026_balanced --parallel --no-related-work

# 3. Run benchmark with RAG calibration
python run_iclr_bench.py 50 3112 --parallel \
  --data-dir iclr2026_unbalanced --calibration calibration.md --no-related-work

# 4. Compute metrics
python metric.py bench_scores.csv
```

## Baselines

Three baselines are provided for comparison against the full multi-agent pipeline. Single-model baselines live under `baselines/`.

```
baselines/
├── always_predict_x/         # trivial baseline: always predicts score=5, Accept
│   └── run_baseline.py
├── direct_review/            # single-turn direct scoring (no review pipeline)
│   └── run_direct_baseline.py
└── structured_review/        # structured review matching merger sections
    ├── run_baseline.py
    └── build_calibration.py
```

### Always-predict-5 baseline (`baselines/always_predict_x/run_baseline.py`)

Predicts score=5 and decision=Accept for every paper. No model calls, no calibration. This is the trivial baseline — any useful system must beat it.

```bash
python baselines/always_predict_x/run_baseline.py 50 3112 --balanced --data-dir iclr2026_unbalanced --calibration calibration.md
```

The `--calibration` flag only excludes calibration paper IDs from sampling — no calibration context is used.

### Direct-scoring baseline (`baselines/direct_review/run_direct_baseline.py`)

Sends the paper directly to the scoring model and asks it to review and score in a single turn. No sub-agent reviews, no merger, no calibration. This measures what the scoring model can do on its own without any pipeline support.

```bash
python baselines/direct_review/run_direct_baseline.py 50 3112 --data-dir iclr2026_unbalanced
```

### Structured-review baseline (`baselines/structured_review/`)

A stronger single-model baseline that outputs the **exact same sections as the multi-agent merger** (Summary, Strengths, Weaknesses, Nice-to-Haves, Novel Insights, Potentially Missed Related Work, Suggestions) but without the agentic pipeline — no harsh/neutral/spark/related-work sub-agents, no merger synthesis.

This isolates the value of the multi-agent pipeline itself: if the full system outperforms this baseline, the improvement comes from multi-perspective synthesis rather than just the review format or scoring prompt.

Key differences from the simple review baseline:
- **Review format**: matches the merger output (7 structured sections vs 4 simple sections)
- **Scoring prompt**: uses the same comparative scoring procedure as the main pipeline
- **Calibration format**: uses `# Final Consolidated Review` headers matching the main pipeline's calibration

```bash
# Build calibration
python baselines/structured_review/build_calibration.py --data-dir iclr2026_unbalanced

# Run benchmark
python baselines/structured_review/run_baseline.py 50 3112 --balanced \
  --data-dir iclr2026_unbalanced --calibration baselines/structured_review/calibration.md
```

### Comparison

| Method | Sub-agents | Review sections | Scoring method | Calibration |
|--------|-----------|-----------------|----------------|-------------|
| Always-predict-5 | None | N/A | Hardcoded 5 | None |
| Direct scoring | None | Brief assessment | Single-turn scoring guide | None |
| Structured review | None | Same 7 sections as merger | Comparative scoring (same as pipeline) | Own (`baselines/structured_review/calibration.md`) |
| Full pipeline | 4 specialized agents | Same 7 sections (via merger) | RAG scoring (Claude Agent SDK) | File-based in `cal/` |

## Calibration

The score predictor tends to overestimate scores. To fix this, we build a **calibration set** stored as individual file pairs in the `cal/` directory:

1. **Sample** papers across score bins (10 per bin)
2. **Run the full review stack** (harsh critic, neutral, spark, related work, merger) on each calibration paper
3. **Save** each as two files in `cal/`:
   - `{title}_paper.md` — the original paper text
   - `{title}_review.md` — all agent outputs + merger + human scores
4. **At scoring time**, the Claude Agent SDK scorer uses Grep/Read tools to dynamically find relevant calibration examples

The scorer agent searches both `_paper.md` and `_review.md` files for keywords related to the paper's topic and quality, then reads the `_review.md` files to see the human scores. This RAG approach avoids the token cost and context window limits of injecting all calibration examples into a single prompt.

Calibration papers are excluded from the benchmark and baseline comparison set via `calibration_ids.json`.

## Dataset: ICLR 2026

Papers are fetched from OpenReview via authenticated API (`fetch_iclr2026.py`):
- Downloads PDFs using `client.get_pdf()` (auth required — OpenReview blocks anonymous downloads)
- Converts to markdown using `pymupdf4llm` with cleanup (strips line numbers, OCR artifacts, review headers)
- Withdrawn papers are kept and treated as Reject (they often have low scores that improve distribution coverage)
- Supports `--balanced` stratified sampling across score bins
- Supports `--data-dir` to output to a custom directory

### Evaluation methodology: balanced calibration, unbalanced testing

We use **balanced (stratified) sampling for calibration** and **unbalanced (natural distribution) sampling for testing**. This is the fairest setup:

- **Balanced calibration** ensures the scoring model sees anchor examples across the full score range (not just the 4-6 cluster that dominates the natural distribution), giving it well-distributed reference points.
- **Unbalanced testing** reflects the real-world score distribution, where ~70% of papers fall in the borderline 4-6 range. This prevents inflated correlation metrics that arise from balanced test sets, where the easy-to-predict extreme papers dominate the signal.

On the same model, same parser, and same year of data, balanced test sampling inflates Spearman from ~0.42 to ~0.70 purely because extreme-score papers are trivially easy to rank. MAE remains nearly identical (~2.4), confirming the model's actual scoring ability is the same — only the metric is inflated.

**Important leakage warning:** some source PDFs contain venue-status headers such as `Published as a conference paper at ICLR 2026`, which directly reveal acceptance status. The current pipeline removes both `Under review ...` and `Published as ...` status headers during text cleanup.

More broadly, this is a general caution for any paper-review benchmark: metadata leakage can enter through parsed PDFs, repository mirrors, camera-ready headers, publication notices, or other artifacts that are not part of the original blind submission. Such leakage may not always produce obviously inflated accuracy, but it can still distort benchmarking results. Future benchmarks should explicitly audit and sanitize these signals before evaluation.

## Metrics

`metric.py` computes:
- **Spearman correlation** (raw and rounded to ICLR scale)
- **Pearson correlation**
- **MAE** (Mean Absolute Error)
- **Weighted MAE** (weighted by inverse frequency of GT score bins — prevents the dominant 4-6 cluster from hiding errors on extreme scores)
- **Bias** (`mean(predicted_score - human_avg_score)`) to measure systematic under-scoring or over-scoring
- **Decision accuracy** (Accept/Reject match — currently N/A since decision prediction is disabled)
- **AUROC** (predicted score as discriminator for Accept vs Reject)
- **AUPRC** (area under precision-recall curve)
- **Optimal threshold** via Youden's J statistic
- **Borderline performance** (papers with GT avg 4-6)
- **Human match** (rounded prediction matches any individual reviewer)
- **Human baselines**:
  - **One-vs-rest**: leave-one-reviewer-out correlation (how well can one reviewer predict another's score?)
  - **Split-half**: repeated random splits of reviewer scores (estimates human inter-rater reliability)

For this project, **Pearson, MAE, bias, and decision quality are more important than Spearman**. Rank correlation is reported as a secondary metric, but it is highly sensitive to small local perturbations, especially in the borderline region where both human scores and accept/reject outcomes are inherently noisy. Since the main goal is calibrated scoring rather than exact global ranking, Pearson and bias are more informative about whether the model is using the same score scale as human reviewers.

Generates a 6-panel plot: raw scatter, human one-vs-rest baseline, human split-half baseline, ROC curve, precision-recall curve, and score bin breakdown.

## LLM-as-Judge

`judge/llm_as_judge.py` compares two AI-generated reviews against human reviews fetched from OpenReview. Uses the Claude Agent SDK to judge which candidate review is better aligned with the human review.

```bash
python judge/llm_as_judge.py <paper_id> <review1_path> <review2_path> [--dataset unbalanced|balanced]
```

## Output Files

| File | Description |
|------|-------------|
| `bench_results.md` | Full reviews for each paper (written incrementally) |
| `bench_scores.csv` | Per-paper: predicted score, GT avg score, all GT reviewer scores, match |
| `bench_scores_scatter.png` | Scatter plot + ROC curve |
| `bench_reviews/` | Individual review files per paper |
| `bench_run.log` | Complete stdout/stderr log of the run |
| `cal/` | Calibration file pairs (`*_paper.md` + `*_review.md`) for RAG scoring |
| `calibration.md` | Calibration examples in single-file format (fallback if no `cal/` dir) |
| `calibration_ids.json` | Paper IDs excluded from benchmark |
| `baselines/always_predict_x/baseline_scores.csv` | Always-predict-5 baseline results |
| `baselines/direct_review/` | Direct-scoring baseline results |
| `baselines/structured_review/scores.csv` | Structured-review baseline results |
| `baselines/structured_review/calibration.md` | Calibration for structured review baseline |
| `baselines/structured_review/calibration_ids.json` | Paper IDs excluded from structured review baseline |
| `webui_runs/` | Reviews generated via the Gradio web UI |
| `scorer_debug.log` | Full scorer agent output for debugging |

## CLI Reference

### `paper_reviewer.py`

```
python paper_reviewer.py <paper.txt> [options]

  --sequential          Run agents sequentially (parallel is the default)
  --with-related-work   Enable related work search & filter (disabled by default)
  --no-spark            Skip spark finder
  --no-neutral          Skip neutral reviewer
  --venue <name>        Set venue (ICLR, NeurIPS, ICML, etc.)
  --calibration <path>  Calibration file/path (default: calibration.md if present)
```

### `run_iclr_bench.py`

```
python run_iclr_bench.py [n] [seed] [options]

  --parallel              Run reviewers concurrently (within each paper)
  --balanced              Stratified sampling across score bins
  --data-dir <path>       Dataset directory (default: AI-Scientist/review_iclr_bench)
  --calibration <path>    Calibration file for RAG score prediction
  --no-related-work       Skip related work agents
  --no-spark              Skip spark finder
  --no-neutral            Skip neutral reviewer
```

### `build_calibration.py`

```
python build_calibration.py [options]

  --data-dir <path>       Dataset directory
  --parallel              Run review agents concurrently
  --no-spark              Skip spark finder
  --no-related-work       Skip related work search
  --no-neutral            Skip neutral reviewer
```

### `baselines/always_predict_x/run_baseline.py`

```
python baselines/always_predict_x/run_baseline.py [n] [seed] [options]

  --balanced              Stratified sampling across score bins
  --data-dir <path>       Dataset directory
  --calibration <path>    Calibration file; excludes calibration IDs
```

### `baselines/direct_review/run_direct_baseline.py`

```
python baselines/direct_review/run_direct_baseline.py [n] [seed] [options]

  --balanced              Stratified sampling across score bins
  --data-dir <path>       Dataset directory
```

### `baselines/structured_review/run_baseline.py`

```
python baselines/structured_review/run_baseline.py [n] [seed] [options]

  --balanced              Stratified sampling across score bins
  --data-dir <path>       Dataset directory
  --calibration <path>    Calibration file (baselines/structured_review/calibration.md)
```

### `baselines/structured_review/build_calibration.py`

```
python baselines/structured_review/build_calibration.py [options]

  --data-dir <path>       Dataset directory
```

### `fetch_iclr2026.py`

```
python fetch_iclr2026.py [n] [seed] [options]

  --balanced        Stratified sampling across score bins
  --data-dir <path> Output directory (default: iclr2026_data)

Requires OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in .env
```

### `judge/llm_as_judge.py`

```
python judge/llm_as_judge.py <paper_id> <review1_path> <review2_path> [--dataset unbalanced|balanced]

Compares two candidate reviews against human reviews from OpenReview.
Uses Claude Agent SDK as the judge.
```

## Cost

The harsh critic and scorer use `claude-sonnet-4-6` via the Claude Agent SDK (cost not exposed by SDK). Neutral, spark, related work, and merger use Qwen and GLM-5 models via OpenRouter. Score parsing uses `gpt-5.4-nano` via OpenRouter. Exact cost depends on current pricing, paper length, output length, and how many calibration files the scorer reads.
