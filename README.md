# Multi-Agent Paper Reviewer

An automated academic paper review system that uses multiple LLM agents with different roles to produce a consolidated review with a score and accept/reject decision. Benchmarked against ICLR 2025 ground truth reviews.

## Architecture

```
Phase 1 (all parallel via OpenRouter chat completions):
  ├── Critical Reviewer ──── minimax/minimax-m2.7
  ├── Neutral Reviewer ───── minimax/minimax-m2.7
  ├── Spark Finder ───────── minimax/minimax-m2.7
  └── Related Work Scout ─── minimax/minimax-m2.7:online → minimax/minimax-m2.7 filter

Phase 2:
  └── Merger ──────────────── minimax/minimax-m2.7

Phase 3:
  └── Score Predictor ─────── minimax/minimax-m2.7
      (optionally calibrated with few-shot examples)
```

**Agents:**

| Agent | Model | Role |
|-------|-------|------|
| Critical Reviewer | `minimax/minimax-m2.7` | Section-by-section rubric review — raises genuine concerns, not nitpicks |
| Neutral Reviewer | `minimax/minimax-m2.7` | Balanced assessment with strengths, weaknesses, novelty |
| Spark Finder | `minimax/minimax-m2.7` | Identifies missing experiments, deeper analysis, untapped applications |
| Related Work Scout | `minimax/minimax-m2.7:online` | Proposes potentially missed references using OpenRouter's online model variant |
| Related Work Filter | `minimax/minimax-m2.7` | Removes already-cited and loosely related results |
| Merger | `minimax/minimax-m2.7` | Synthesizes all inputs into a final structured review with subscores |
| Score Predictor | `minimax/minimax-m2.7` | Predicts a continuous score from the multi-agent reviews |

All calls now go through OpenRouter using chat completions. Structured outputs for merger and score prediction are still enforced, and reasoning configuration is passed through `extra_body` where supported.

## Review And Scoring Design

The merger acts as an area chair and applies several filters:

- **Nonsense filter**: Removes criticisms that are factually wrong or misunderstand the paper
- **Nitpick filter**: Drops formatting, style, and minor phrasing complaints
- **Scope check**: For every weakness, asks:
  1. Is this a *real* weakness or a *nice-to-have*?
  2. Could this omission be *intentional* (scope decision, space constraint)?
  3. Is this within the paper's stated scope?
- **Calibration**: Uses score distribution priors (~5% strong accept, ~40% borderline reject, ~30% clear reject) to prevent score inflation
- **Few-shot calibration** (optional): Injects real examples of multi-agent review bundles paired with human scores/decisions into the score predictor

Real weaknesses go in `"weaknesses"` and inform the final assessment. Nice-to-haves go in `"nice_to_haves"` and should not be treated like core flaws. The merger outputs structured JSON; the separate score predictor outputs a **continuous score from 1.0 to 10.0** using the review bundle, with optional calibration examples.

## Motivation

Prior academic review agents have several practical problems that this project tries to avoid.

CSPaper Review (CSPR) appears to rely on a forced-score style of review generation. It force the AI to generate a review for every single score in the score range (from 1-10) and then merge them. In practice, this design can encourage overly picky, internally inconsistent, or weakly grounded criticism, and may lead to contradictions across different parts of the review. Its related-work stage also appears to have relatively low precision, introducing a substantial amount of noisy or only weakly relevant feedback. Such noise can depress scores artificially and reduce the practical usability of the system for authors.

> Review agents: For each valid rating/score level defined by the target conference (e.g., 1-strong reject to 5-strong accept), we force a dedicated agent to (concurrently) generate reviews that strictly justify the assigned score/rating. A review selector identifies three most realistic reviews: best justified, more optimistic, and more critical. They are synthesized into a coherent output primarily based on the best-justified review but selectively incorporating insights from the other two versions. Finally, a calibration step ensures coherence between overall and sub-dimensional scores (e.g., novelty, clarity), ensuring a well-aligned and balanced final review.

CSPR's calibration approach also appears to depend heavily on semantic analysis of generated critiques. This is potentially problematic because the number of negative points raised in a review is not a reliable proxy for overall paper quality. A paper may elicit many minor comments without having serious flaws, while a substantially stronger paper may have only a few high-impact concerns. As a result, direct semantic aggregation of negative points can distort score calibration.

A broader issue in this literature is evaluation methodology. Many systems emphasize MAE or exact agreement with a human score, but these metrics can be misleading on imbalanced datasets. For example, on a non-balanced conference sample, an always-predict-6 baseline can already achieve deceptively strong performance. Without careful stratification, such evaluation protocols can overstate model quality. CSPR also appears to exhibit selection bias in paper collection, which further limits confidence in the reported results.

The Stanford review agent appears to encounter similar related-work precision issues. It also exhibits a failure mode in which the model may incorrectly treat papers, methods, or models outside its training timeline as fabricated and challenge the user on that basis. Its reported correlation results are also difficult to interpret, since the presentation partly relies on human-human agreement rather than a cleaner AI-to-human benchmark. More broadly, like CSPR, it is neither open source nor accompanied by a sufficiently detailed technical description, which makes the system difficult to inspect, explain, or validate.

## Quick Start

```bash
# Clone
git clone https://github.com/weathon/review_agent.git
cd review_agent

# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter API key
echo 'OPENROUTER_API_KEY="sk-or-..."' > .env

# For fetching ICLR 2025 papers, also add OpenReview credentials:
echo 'OPENREVIEW_USERNAME="your@email.com"' >> .env
echo 'OPENREVIEW_PASSWORD="yourpassword"' >> .env
```

## Usage

### Review a single paper

```bash
python paper_reviewer.py paper.txt --parallel --venue NeurIPS
python paper_reviewer.py paper.txt --parallel --no-related-work --no-spark
```

### Run the local Web UI

```bash
./run_webui.sh
```

Then open `http://127.0.0.1:7860`.

The UI is intentionally simple:
- no auth
- BYOK via an OpenRouter API key field in the page
- upload a `.pdf` / `.txt` / `.md` paper file, or paste paper text directly
- PDF parsing reuses the dataset builder parser from `fetch_iclr2025.py`
- toggles for parallel mode, related work, spark finder, and calibration

### Run everything (fetch → calibrate → benchmark)

```bash
./run_all.sh
```

This runs the full pipeline:
1. Fetches 200 ICLR 2025 papers (balanced sampling, includes withdrawn)
2. Builds calibration set (~10 papers, multi-agent review bundles paired with human scores)
3. Runs always-predict-6 baseline
4. Runs full reviewer benchmark with calibration (50 papers, 3 concurrent)
5. Computes metrics + generates plots

### Step by step

```bash
# 1. Fetch ICLR 2025 papers with full text
python fetch_iclr2025.py 200 42 --balanced

# 2. Build calibration set
python build_calibration.py --data-dir iclr2025_data --parallel

# 3. Run baseline
python run_baseline.py 50 4112 --balanced --data-dir iclr2025_data --calibration calibration.md

# 4. Run benchmark with calibration
python run_iclr_bench.py 50 4112 --parallel --balanced \
  --data-dir iclr2025_data --calibration calibration.md

# 5. Compute metrics
python metric.py bench_scores.csv
```

## Calibration

The score predictor tends to overestimate scores. To fix this, we build a **calibration set**:

1. **Sample** 1 paper per score bin (+ extra from borderline bins 5 and 6)
2. **Run the full review stack** (critic, neutral, spark, related work, merger) on each calibration paper
3. **Pair** the resulting review bundle with real human reviewer scores and decisions
4. **Save** as `calibration.md` — injected into the score predictor prompt as few-shot examples

This shows the score predictor what "a paper that humans scored 3" vs "a paper that humans scored 8" looks like in terms of the assembled review bundle.

Calibration papers are excluded from both the benchmark and the baseline comparison set via `calibration_ids.json`.

## Dataset: ICLR 2025

Papers are fetched from OpenReview via authenticated API (`fetch_iclr2025.py`):
- Downloads PDFs using `client.get_pdf()` (auth required — OpenReview blocks anonymous downloads)
- Converts to markdown using `pymupdf4llm` with cleanup (strips line numbers, OCR artifacts, review headers)
- Withdrawn papers are kept and treated as Reject (they often have low scores that improve distribution coverage)
- Supports `--balanced` stratified sampling across score bins

**Important leakage warning:** some source PDFs contain venue-status headers such as `Published as a conference paper at ICLR 2025`, which directly reveal acceptance status. The current pipeline removes both `Under review ...` and `Published as ...` status headers during text cleanup. For already-generated local datasets, run `python fix_paper_headers.py` before benchmarking.

This leakage does not appear to fully explain the current system behavior, since the model still shows substantial under-scoring and relatively weak score calibration. In other words, the dominant error mode is still conservative scoring rather than obvious label copying. However, venue-status headers can still bias evaluation in subtle ways, especially for decision-related metrics, and should therefore be removed.

More broadly, this is a general caution for any paper-review benchmark: metadata leakage can enter through parsed PDFs, repository mirrors, camera-ready headers, publication notices, or other artifacts that are not part of the original blind submission. Such leakage may not always produce obviously inflated accuracy, but it can still distort benchmarking results. Future benchmarks should explicitly audit and sanitize these signals before evaluation.

Score distribution (ICLR 2025 reviewer ratings: 1, 3, 5, 6, 8, 10):

| Bin | Accept | Reject | Total |
|-----|--------|--------|-------|
| ~3  | 0      | 3      | 3     |
| ~4  | 0      | 9      | 9     |
| ~5  | 3      | 21     | 24    |
| ~6  | 24     | 21     | 45    |
| ~7  | 13     | 0      | 13    |
| ~8  | 6      | 0      | 6     |

**Use `--balanced` for evaluation** — random sampling oversamples bin 6 (45% of data) and inflates baseline performance.

## Metrics

`metric.py` computes:
- **Spearman correlation** (raw and rounded to ICLR scale)
- **Pearson correlation**
- **MAE** (Mean Absolute Error)
- **Bias** (`mean(predicted_score - human_avg_score)`) to measure systematic under-scoring or over-scoring
- **Decision accuracy** (Accept/Reject match)
- **AUROC** (predicted score as discriminator for Accept vs Reject)
- **Optimal threshold** via Youden's J statistic
- **Borderline performance** (papers with GT avg 4-6)
- **Human match** (rounded prediction matches any individual reviewer)

For this project, **Pearson, MAE, bias, and decision quality are more important than Spearman**. Rank correlation is reported as a secondary metric, but it is highly sensitive to small local perturbations, especially in the borderline region where both human scores and accept/reject outcomes are inherently noisy. Since the main goal is calibrated scoring rather than exact global ranking, Pearson and bias are more informative about whether the model is using the same score scale as human reviewers.

Generates a 3-panel plot: raw scatter, rounded scatter, and ROC curve.

## Output Files

| File | Description |
|------|-------------|
| `bench_results.md` | Full reviews for each paper (written incrementally) |
| `bench_scores.csv` | Per-paper: predicted score, GT avg score, all GT reviewer scores, match |
| `bench_scores_scatter.png` | Scatter plot + ROC curve |
| `bench_run.log` | Complete stdout/stderr log of the run |
| `baseline_scores.csv` | Baseline results |
| `calibration.md` | Few-shot calibration examples (review bundle + human scores) |
| `calibration_ids.json` | Paper IDs excluded from benchmark |

## CLI Reference

### `paper_reviewer.py`

```
python paper_reviewer.py <paper.txt> [options]

  --parallel          Run all reviewers concurrently
  --no-related-work   Skip related work search + filter
  --no-spark          Skip spark finder
  --venue <name>      Set venue (ICLR, NeurIPS, ICML, etc.)
```

### `run_iclr_bench.py`

```
python run_iclr_bench.py [n] [seed] [options]

  --parallel              Run reviewers concurrently (within each paper)
  --balanced              Stratified sampling across score bins
  --data-dir <path>       Dataset directory (default: AI-Scientist/review_iclr_bench)
  --calibration <path>    Calibration file for few-shot score prediction
  --no-related-work       Skip related work agents
  --no-spark              Skip spark finder
```

### `build_calibration.py`

```
python build_calibration.py [seed] [options]

  --data-dir <path>       Dataset directory
  --parallel              Run review agents concurrently
  --no-spark              Skip spark finder
  --no-related-work       Skip related work search
```

### `run_baseline.py`

```
python run_baseline.py [n] [seed] [options]

  --balanced              Stratified sampling across score bins
  --data-dir <path>       Dataset directory
  --calibration <path>    Calibration file; excludes calibration IDs
```

### `fetch_iclr2025.py`

```
python fetch_iclr2025.py [n] [seed] [options]

  --balanced    Stratified sampling across score bins

Requires OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in .env
```

## Cost

All API calls now go through OpenRouter using `minimax/minimax-m2.7`, with `minimax/minimax-m2.7:online` used for the related-work search stage. Exact cost depends on current OpenRouter pricing, paper length, output length, and online-search usage. Reusing a single model family across all stages simplifies deployment and reduces score shifts caused by mixing providers.
