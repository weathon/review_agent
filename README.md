# Multi-Agent Paper Reviewer

An automated academic paper review system that uses multiple LLM agents with different roles to produce a consolidated review with a score and accept/reject decision. Benchmarked against ICLR 2022 ground truth reviews.

## Architecture

```
Stage 1 (parallel):
  ├── Critical Reviewer ──── Claude Sonnet 4.6 (Claude Code SDK, free)
  ├── Neutral Reviewer ───── GLM-5 (OpenRouter)
  └── Related Work Scout ─── Perplexity sonar-pro → GLM-5 filter (OpenRouter)

Stage 2:
  └── Spark Finder ────────── Claude Sonnet 4.6 (Claude Code SDK, free)

Stage 3:
  └── Merger ──────────────── Claude Sonnet 4.6 (Claude Code SDK, free)
```

**Agents:**

| Agent | Model | Role |
|-------|-------|------|
| Critical Reviewer | Claude Sonnet 4.6 | Raises genuine concerns, questions, and blind spots — not nitpicking |
| Neutral Reviewer | GLM-5 | Balanced assessment with strengths, weaknesses, novelty |
| Spark Finder | Claude Sonnet 4.6 | Finds novel ideas, unexpected connections, creative extensions |
| Related Work Scout | Perplexity sonar-pro | Web-grounded search for potentially missed references |
| Related Work Filter | GLM-5 | Removes already-cited and loosely related results |
| Merger | Claude Sonnet 4.6 | Synthesizes all inputs into a final JSON review with score + decision |

Claude calls go through the Claude Code SDK (free, no API cost). Everything else goes through OpenRouter.

## Merger Design

The merger acts as an area chair and applies several filters:

- **Nonsense filter**: Removes criticisms that don't exist in the paper or misunderstand the contribution
- **Nitpick filter**: Drops formatting, style, and minor phrasing complaints
- **Scope check**: For every weakness, asks:
  1. Is this a *real* weakness or a *nice-to-have*?
  2. Could this omission be *intentional* (scope decision, space constraint)?
  3. Is this within the paper's stated scope?
- Real weaknesses go in `"weaknesses"` and affect the score
- Nice-to-haves go in `"nice_to_haves"` and do NOT affect the score
- Potentially missed related work is informational only, never penalized

The merger outputs a structured JSON with a **continuous score from 1.0 to 10.0** and is explicitly prompted to use the full range (not cluster around 5-6).

## Setup

```bash
# Clone
git clone https://github.com/weathon/review_agent.git
cd review_agent

# Install dependencies
pip install openai python-dotenv claude-code-sdk

# Dataset is loaded automatically from HuggingFace (davidheineman/iclr-2026)
pip install datasets

# Set your OpenRouter API key
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > .env
```

## Usage

### Review a single paper

```bash
# Basic (sequential)
python paper_reviewer.py paper.txt

# Parallel OpenRouter agents + specify venue
python paper_reviewer.py paper.txt --parallel --venue NeurIPS

# Skip optional agents
python paper_reviewer.py paper.txt --parallel --no-related-work --no-spark
```

### Run the ICLR 2026 benchmark

Uses `davidheineman/iclr-2026` from HuggingFace (17K+ papers with abstracts and reviewer scores, no final decisions yet — accept/reject is inferred from average score >= 5.5).

```bash
# 10 papers, random sampling
python run_iclr_bench.py 10 42 --parallel

# 100 papers, balanced sampling (recommended for evaluation)
python run_iclr_bench.py 100 4112 --parallel --balanced

# Minimal (no spark, no related work — just critic + neutral + merger)
python run_iclr_bench.py 10 42 --parallel --no-spark --no-related-work
```

### Run the baselines

```bash
# Always-predict-6 baseline
python run_baseline.py 100 4112

# Balanced sampling baseline
python run_baseline.py 100 4112 --balanced
```

## Sampling Methods

The ICLR 2026 dataset (17K+ papers) has a skewed score distribution (reviewer ratings 0-10, even numbers only):

```
Score ~2:  13 papers    ██
Score ~3:  16 papers    ███
Score ~4:  83 papers    █████████████████
Score ~5: 130 papers    ██████████████████████████
Score ~6: 179 papers    ████████████████████████████████████
Score ~7:  51 papers    ██████████
Score ~8:  27 papers    █████
Score ~9:   1 paper     ▏
```

Most papers cluster around 5-6 (borderline). This matters because:

### Random sampling (`python run_iclr_bench.py 100 4112`)

Draws papers uniformly at random. The sample reflects the dataset's natural distribution — ~68% rejects, ~32% accepts, concentrated near the decision boundary. The **always-predict-Accept baseline gets 32% accuracy** because most papers are rejects that it misclassifies.

The always-predict-6 baseline achieves only **1.04 avg score difference** because the mean GT score is 5.33 — most papers are close to 6 anyway. This makes it look deceptively good on score prediction but terrible on decision accuracy.

### Balanced sampling (`--balanced`)

Stratified sampling that draws **equal numbers from each score bin** (rounded to nearest integer). This ensures low-scoring (2-3) and high-scoring (8-9) papers are represented proportionally, not drowned out by the 5-6 cluster.

With balanced sampling:
- The always-predict-6 baseline drops to **39.3% accuracy**
- Average score difference jumps to **1.86** (the constant guess is now much worse)
- Per-bin accuracy reveals the baseline is 0% on score 2-3 papers and 100% on score 8-9

**Use `--balanced` for evaluation.** Random sampling inflates baseline performance and makes it hard to tell if the model is actually discriminating.

## Output Files

| File | Description |
|------|-------------|
| `bench_results.md` | Full reviews for each paper (written incrementally) |
| `bench_scores.csv` | Per-paper: predicted score, GT avg score, all GT reviewer scores, match |
| `bench_run.log` | Complete stdout/stderr log of the run |
| `baseline_scores.csv` | Baseline results (random sampling) |
| `*_review.md` | Individual paper review output |

## CLI Flags

### `paper_reviewer.py`

| Flag | Description |
|------|-------------|
| `--parallel` | Run OpenRouter agents concurrently with Claude agents |
| `--no-related-work` | Skip related work search + filter (saves 2 API calls) |
| `--no-spark` | Skip spark finder (saves 1 Claude call) |
| `--venue <name>` | Set venue (e.g. ICLR, NeurIPS, ICML) — prompts become venue-specific |

### `run_iclr_bench.py`

| Flag | Description |
|------|-------------|
| `--parallel` | Run OpenRouter agents concurrently |
| `--balanced` | Use stratified sampling instead of random |
| `--no-related-work` | Skip related work agents |
| `--no-spark` | Skip spark finder |

## Cost

Claude SDK calls are free (uses your local Claude Code installation). OpenRouter costs depend on the models used — with GLM-5 for neutral review and Perplexity sonar-pro for related work, expect ~$0.01-0.05 per paper.
