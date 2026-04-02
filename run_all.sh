#!/bin/bash
set -e

echo "============================================"
echo "Full Pipeline: Fetch → Calibrate → Benchmark"
echo "============================================"

# # ── Step 1: Re-fetch dataset (with withdrawn papers included) ──
# echo ""
# echo ">>> Step 1: Fetching ICLR 2025 papers (200, balanced, includes withdrawn)"
# echo "    This re-uses cached API notes but re-samples and re-downloads PDFs."
# rm -f iclr2025_data/ratings.csv  # force re-sample
# python fetch_iclr2025.py 200 42 --balanced

# ── Step 2: Check dataset distribution ──
echo ""
echo ">>> Step 2: Dataset distribution check"
python -c "
import csv
from collections import Counter
with open('iclr2025_data/ratings.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total papers: {len(rows)}')
decs = Counter(r['gt_binary'] for r in rows)
for k, v in decs.most_common():
    print(f'  {k}: {v}')
bins = Counter(round(float(r['avg_score'])) for r in rows)
print('Score bins:')
for k in sorted(bins):
    print(f'  ~{k}: {bins[k]}')
"

# ── Step 3: Build calibration set ──
echo ""
echo ">>> Step 3: Building calibration set (sub-agents only, no merger)"
python build_calibration.py --data-dir iclr2025_data --parallel --no-related-work

# ── Step 4: Run baseline ──
# echo ""
# echo ">>> Step 4: Running baseline (always predict 6)"
# python run_baseline.py 50 4112 --data-dir iclr2025_data --calibration calibration.md --no-related-work# --balanced 

# ── Step 5: Run benchmark with calibration ──
echo ""
echo ">>> Step 5: Running benchmark (50 papers, balanced, with calibration)"
python run_iclr_bench.py 100 3112 --parallel --data-dir iclr2025_data --calibration calibration.md --no-related-work

# ── Step 6: Compute metrics ──
echo ""
echo ">>> Step 6: Computing metrics"
python metric.py bench_scores.csv

echo ""
echo "============================================"
echo "Done! Output files:"
echo "  iclr2025_data/ratings.csv    - dataset"
echo "  calibration.md               - calibration examples"
echo "  calibration_ids.json         - excluded paper IDs"
echo "  bench_results.md             - full reviews"
echo "  bench_scores.csv             - predictions vs GT"
echo "  bench_run.log                - full log"
echo "  baseline_scores.csv          - baseline results"
echo "  bench_scores_scatter.png     - scatter plot + ROC"
echo "============================================"
