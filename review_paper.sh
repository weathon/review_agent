#!/usr/bin/env bash
# Review a paper. Each step is one agent call. All data passes through files in $WORK.
#
# Usage: bash review_paper.sh paper.txt [work_dir]
#
# Config is at the top. Disable agents by commenting out their line.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER="$(realpath "$1")"
WORK="${2:-$(mktemp -d /tmp/review_XXXX)}"
mkdir -p "$WORK"
echo "paper=$PAPER  work=$WORK"

# ── Config ──
VENUE="ICLR"
BASE="deepseek/deepseek-v3.2"
MODEL_HARSH="claude:claude-opus-4-6"
MODEL_NEUTRAL="$BASE"
MODEL_SPARK="$BASE"
MODEL_MERGER="$BASE"
CAL_DIR="$SCRIPT_DIR/cal"

# ── Phase 1: reviewers (parallel) ──
python "$SCRIPT_DIR/agents/reviewer.py" --paper "$PAPER" --prompt "$SCRIPT_DIR/prompts/harsh_critic.txt"    --model "$MODEL_HARSH"   --output "$WORK/harsh.txt"   --venue "$VENUE" --name harsh   &
python "$SCRIPT_DIR/agents/reviewer.py" --paper "$PAPER" --prompt "$SCRIPT_DIR/prompts/neutral_reviewer.txt" --model "$MODEL_NEUTRAL" --output "$WORK/neutral.txt" --venue "$VENUE" --name neutral &
python "$SCRIPT_DIR/agents/reviewer.py" --paper "$PAPER" --prompt "$SCRIPT_DIR/prompts/spark_finder.txt"    --model "$MODEL_SPARK"   --output "$WORK/spark.txt"   --venue "$VENUE" --name spark   &
wait

# ── Phase 2: merger ──
python "$SCRIPT_DIR/agents/merger.py" --paper "$PAPER" --harsh "$WORK/harsh.txt" --neutral "$WORK/neutral.txt" --spark "$WORK/spark.txt" --model "$MODEL_MERGER" --output "$WORK/merged.txt"

# ── Phase 3: scorer ──
python "$SCRIPT_DIR/agents/scorer.py" --review "$WORK/merged.txt" --paper "$PAPER" --cal-dir "$CAL_DIR" --output "$WORK/score.txt"

echo ""; echo "Score: $(cat "$WORK/score.txt")"; echo "Review: $WORK/merged.txt"
