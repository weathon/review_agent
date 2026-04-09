#!/bin/bash
set -e

echo "============================================"
echo "Full Pipeline: Fetch → Calibrate → Benchmark"
echo "============================================"

# # ── Step 1: Re-fetch dataset (with withdrawn papers included) ──
# echo ""
# echo ">>> Step 1: Fetching ICLR 2025 papers (200, balanced, includes withdrawn)"
# echo "    This re-uses cached API notes but re-samples and re-downloads PDFs."
# rm -f iclr2026_balanced/ratings.csv  # force re-sample
# python fetch_iclr2025.py 200 42 --balanced

# ── Step 2: Check dataset distribution ── 

# set -a && source "$(dirname "$0")/.env" && set +a

# export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
# export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
# export ANTHROPIC_API_KEY="" # Important: Must be explicitly empty

# export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen/qwen3.6-plus"
# export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen/qwen3.5-flash-02-23"
# export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen/qwen3.5-flash-02-23"
# export CLAUDE_CODE_SUBAGENT_MODEL="qwen/qwen3.6-plus"



echo ""
echo ">>> Step 2: Dataset distribution check"
python -c "
import csv
from collections import Counter
with open('iclr2026_balanced/ratings.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total papers: {len(rows)}')
decs = Counter(r['gt_binary'] for r in rows)
for k, v in decs.most_common():
    print(f'  {k}: {v}')
bins = Counter(round(float(r['avg_score'])) for r in rows)
print('Score bins:')
for k in sorted(bins):
    print(f'{k}: {bins[k]/len(rows):.02f}')
"

# ── Step 3: Build calibration set ──
# echo ""
# echo ">>> Step 3: Building calibration set (sub-agents only, no merger)"
# python build_calibration.py --data-dir iclr2026_balanced --parallel --no-related-work # --no-neutral


# ── Step 5: Run benchmark with calibration ──
echo ""
rm -rf bench_reviews/
mkdir bench_reviews
python run_iclr_bench.py 50 3 --parallel --data-dir iclr2026_balanced --calibration calibration.md --no-related-work # --no-neutral

# ── Step 6: Compute metrics ──
echo ""
echo ">>> Step 6: Computing metrics"
python metric.py bench_scores.csv

echo ""
echo "============================================"
echo "Done! Output files:"
echo "  iclr2026_balanced/ratings.csv    - dataset"
echo "  calibration.md               - calibration examples"
echo "  calibration_ids.json         - excluded paper IDs"
echo "  bench_results.md             - full reviews"
echo "  bench_scores.csv             - predictions vs GT"
echo "  bench_run.log                - full log"
echo "  baseline_scores.csv          - baseline results"
echo "  bench_scores_scatter.png     - scatter plot + ROC"
echo "============================================"
