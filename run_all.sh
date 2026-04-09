#!/bin/bash
set -e

DATA_DIR="iclr2026_balanced"

echo "============================================"
echo "Full Pipeline: Calibrate → Benchmark"
echo "============================================"

# ── Dataset distribution check ──
echo ""
echo ">>> Dataset distribution"
python3 -c "
import csv
from collections import Counter
with open('${DATA_DIR}/ratings.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total papers: {len(rows)}')
decs = Counter(r['gt_binary'] for r in rows)
for k, v in decs.most_common():
    print(f'  {k}: {v}')
bins = Counter(round(float(r['avg_score'])) for r in rows)
print('Score bins:')
for k in sorted(bins):
    print(f'  {k}: {bins[k]/len(rows):.02f}')
"

# ── Build calibration set ──
echo ""
echo ">>> Building calibration set"
python3 gpt_agent_sdk.py --calibration "$DATA_DIR" --seed 42

# ── Run benchmark ──
echo ""
echo ">>> Running benchmark"
rm -rf bench_reviews/ && mkdir bench_reviews
python3 gpt_agent_sdk.py --benchmark "$DATA_DIR" --n_samples 50 --seed 3

# ── Compute metrics ──
echo ""
echo ">>> Computing metrics"
python3 metric.py bench_scores.csv

echo ""
echo "============================================"
echo "Done! Output files:"
echo "  cal/                     - calibration examples"
echo "  calibration_ids.json     - excluded paper IDs"
echo "  bench_scores.csv         - predictions vs GT"
echo "  bench_reviews/           - individual reviews"
echo "============================================"
