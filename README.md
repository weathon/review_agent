
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


现有的自动化论文审稿系统存在两类核心问题：方法设计上的缺陷和评估方式上的盲区。

**方法缺陷。** CSpaper采用强制评分策略，要求模型为评分范围内的每一个分数生成对应的审稿意见，然后合并。这种设计迫使模型在论文没有严重问题时也必须捏造低分理由，产生大量无意义的批评（例如要求作者定义余弦相似度）。其评分校准依赖对生成文本的情感分析，但负面评论的数量并不能可靠地反映论文质量：一篇优秀的论文可能只有少数高价值问题，而一篇普通的论文可能引发大量琐碎评论。当一篇论文质量较高时，强制低分审稿人找不到真正的问题，只能生成大量琐碎甚至无中生有的批评来填充内容。这些批评在文本上仍然表现为负面语言，而情感分析无法区分"论文确实存在严重缺陷"和"审稿人被迫挑刺"这两种截然不同的情况。结果是，论文越好，强制低分审稿人越需要捏造问题，生成的负面文本越多，情感分析给出的质量评估反而越低。这形成了一个系统性的反向偏差：论文质量与系统评分之间的关系在高分区间被扭曲。Stanford的审稿系统采用子维度评分加线性回归的方式预测最终分数，但由于语言模型固有的讨好倾向，子维度分数往往集中在一个狭窄区间内，缺乏区分度，导致回归输入本身就不具备信息量。此外，该系统在处理超出训练数据时间范围的论文时，可能会将真实存在的模型或方法误判为虚构内容。两个系统均未开源，也缺乏足够详细的技术说明。

我们的方法回避了上述问题。在审稿生成阶段，我们采用多智能体协作架构，由不同视角的子审稿人独立生成意见，再由合并模块进行质量过滤，主动移除无依据或过度挑剔的批评，而非强制为每个分数档位生成意见。在评分阶段，我们不使用情感分析或子维度回归，而是通过检索增强的方式找到相似论文的真实审稿分数作为锚点，让评分建立在横向比较的基础上。

**评估盲区。** OpenReviewer在不平衡的数据集上报告了看似合理的误差指标，但一个总是预测固定分数的简单基线也能达到类似水平，这说明其评估协议可能高估了模型能力。DeepReviewer v2提出了严格问题覆盖率作为核心指标，衡量AI审稿覆盖了多少人类审稿人提出的问题，但这一指标存在结构性缺陷。首先，它只衡量召回率而完全忽略精确率，不对过度生成施加任何惩罚——系统可以通过大量输出来提高覆盖率，无论其中多少批评实际上毫无价值。其次，更根本的问题在于，它将人类审稿意见作为唯一的正确答案。AI系统发现了人类遗漏的真实问题不会获得任何奖励，而AI重复了人类提出的无效批评反而会被计为成功覆盖。此外，该工作使用ICLR 2025的审稿数据进行评估，而该轮审稿中已有相当比例的审稿意见由AI生成，将AI生成的审稿作为参考标准来衡量另一个AI系统的覆盖率，本身就消解了这一评估的意义。这种评估逻辑将目标限定为"与人类对齐"，但自动化审稿的价值不应止步于模仿人类审稿人，而应追求超越人类审稿人的局限性。人类审稿本身就存在遗漏、误读和主观偏差，一个真正有用的系统应当被评估的是它能否准确识别论文中的真实问题并提供有效的改进方向，而非它在多大程度上复制了人类的判断。

ScholarPeer采用LLM-as-judge进行side-by-side评估，在Technical Accuracy、Constructive Value、Analytical Depth、Significance Assessment等维度上比较两份审稿的优劣。这一方案的核心问题在于LLM judge本身对伪形式化内容存在系统性偏好：提出问题数量更多、引用文献更丰富、结构看起来更完整的审稿更容易获得高分，而judge并不会逐条验证这些批评是否真正relevant。换言之，该评估测量的是"看起来像好审稿"而非"是好审稿"。ScholarPeer同时提出了H-Max score，以人类最佳审稿为锚点（5分），评估AI审稿是否提供了超越人类的增量价值。这一思路具有参考价值（比DeepReview v2的recall要好），但缺少有效性过滤：当AI提出了人类未提及的观点时，judge直接给予高分，却未验证该观点本身是否正确且与论文相关。我们在此基础上引入有效性过滤，将评估标准从"AI是否说了人类没说的话"收紧为"AI是否提出了人类遗漏的、且确实存在于论文中的真实问题"。

此外，ScholarPeer提出Review Diversity Score，通过embedding相似度衡量同一系统对同一论文多次生成审稿的文本差异度。然而，diversity本身并非目标——一个系统每次都准确指出同一个核心缺陷，其diversity为零但质量很高；反之，一个不稳定的系统每次给出不同但大多无效的意见，diversity很高但毫无价值。真正有意义的问题不是"系统输出是否多样"，而是"多次生成经过集成后，最终输出的质量是否提高"。我们通过ensemble实验直接回答这一问题：对同一论文进行多次审稿生成，经合并模块过滤集成后，测量最终输出的有效性和无意义率是否优于单次生成，从而将diversity从一个孤立的描述性指标转化为对下游审稿质量的实际贡献度量。

我们提出以审稿意见的有效性作为核心评估维度，同时引入无意义率来惩罚凑数行为。有效性比覆盖率更能反映审稿的实际价值，因为它直接回答了最关键的问题：系统提出的批评是否真实存在于论文中，是否能帮助作者改进。

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
