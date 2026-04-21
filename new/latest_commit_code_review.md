# Code Review: `060cddd4` (`split merger into merger + scorer`)

## Findings

### Medium - The OpenAI scorer ignored the runtime `no_cal` argument

Calibration moved into the scorer in this commit, but on the OpenAI path the scorer was originally constructed once at import time from CLI state rather than from the `run_pipeline(..., no_cal=...)` argument.

That meant the Claude path and OpenAI path disagreed on what actually controlled calibration. The CLI entrypoint usually masked this because `--no_cal` was already present in `sys.argv` during import, but any programmatic caller could get the wrong scorer behavior on the OpenAI path.

## Notes

- Scorer access to `human_reviews` is intentional, and prompt-level guidance is acceptable here; hard capability isolation from the paper is not required for this design.
- Benchmark review markdown not carrying the final score is non-blocking because the score and decision are still persisted in the benchmark CSV.

## Verdict

With the design clarifications above, the split looks conceptually fine. The only correctness issue I found was the OpenAI scorer's import-time `no_cal` binding, and that has been fixed locally in the working tree by constructing the scorer with the per-run `no_cal` value.

## Verification

- Parsed `main.py`, `scorer.py`, and `claude_merger.py` successfully with `python3`/`ast`.
- Did not run an end-to-end review pipeline, so this review is based on static inspection plus the local follow-up fix.
