## Summary
Blueprint-Bench introduces a 50-apartment benchmark for reconstructing 2D floor plans from ~20 interior photos per apartment, intended to probe “spatial intelligence” across LLMs, image generators, and agent scaffolds. It standardizes outputs via a strict drawing specification (9 rules) and scores predictions by extracting a room connectivity graph plus room size ranking, then computing a weighted composite similarity score.

## Strengths
- **Clear, automatable end-to-end pipeline**: The paper specifies a concrete extraction procedure (red-dot detection → flood-fill rooms → door scanning) and a structured similarity score over graphs/room statistics (Sec. 2.3, Fig. 4).
- **Interesting empirical observation (with caveats)**: The results document substantial gaps between a human baseline and current model/agent systems on this multi-image reconstruction setup (Sec. 3, Fig. 7).

## Weaknesses

### Fatal
- **Construct validity is not established; scores confound “spatial reconstruction” with compliance to a brittle rendering DSL**. The benchmark requires strict pixel-level conventions (e.g., “3 pixels wide,” “only pure red/black/white/green,” “no gaps,” fixed dot sizes; Sec. 2.1). The paper explicitly acknowledges that rule violations can prevent the scorer from reflecting intended outputs and that “Blueprint-Bench should test spatial intelligence, not instruction following” (Sec. 2.4), yet the main results attribute very low scores primarily to instruction-following failures (“cannot be scored by our algorithm”; Sec. 3). As a result, the headline comparative claim (“models are at/below random on spatial intelligence”) is not supported without additional validation showing the metric tracks spatial correctness across heterogeneous model families.

### Major
- **Room identity defined by size-rank entangles topology and area and can mis-score correct connectivity**. The extractor “assign[s] unique IDs based on their size rank” (Sec. 2.3), and the limitations section concedes that size-ranking mistakes “cause additional penalties when scoring the connectivity” (Sec. 2.4). The human analysis further confirms this failure mode: humans “drawn such that the connectivity … was correct” but “did not always get the size ranking correct,” causing “a harsh penalty,” and the authors suspect the metric underestimates the human lead (Sec. 3). This directly undermines the interpretation of absolute scores and cross-model comparisons.
- **The “random baseline” is not random and is insufficiently specified, making “at/below random” hard to interpret**. The paper defines a “worst-case baseline by generating typical floor plans … without any image input” (Sec. 2.2), but provides no sampling procedure/distributional definition. Since the score includes room/door counts and graph density (Sec. 2.3), a “typical-plan prior” can score non-trivially without being “random,” weakening the central rhetorical claim.
- **Ground-truth creation is under-specified for a 50-example benchmark**. Ground truth is “adapted from the apartment listing’s official floor plan image” to satisfy the 9 rules (Sec. 2.1), but the paper does not describe the adaptation protocol, annotator process, or how ambiguities/omissions are handled. Given the metric’s sensitivity (Sec. 2.4) and small dataset size, this is a serious threat to benchmark interpretability.

### Minor
- **Statistical claims are not adequately supported in the writeup**. The paper states several models “statistically perform better than the random baseline” (Sec. 3) but does not specify the statistical test, unit of analysis (apartments vs. epochs), or provide p-values/effect sizes.
- **Human baseline is computed on only a 12-apartment subset** (“This data is from a subset … (12 instead of 50)”; Fig. 7 caption text), which makes the human–model gap harder to interpret and potentially sensitive to subset selection.

### Trivial
None.

## Nice-to-Haves
- Add a **metric validity study**: e.g., human pairwise judgments of which prediction is closer to ground truth and correlation with the proposed score, plus a breakdown separating (i) connectivity accuracy, (ii) size/area accuracy, and (iii) format compliance.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **“LLM SVG rasterization/antialiasing introduces extra failure modes”**: The paper indeed uses SVG→image for LLMs (Sec. 2.2), but the critic’s specific antialiasing/pixel-purity concerns are speculative without evidence the authors failed to control rendering, and the paper’s main validity issues already cover format brittleness more directly.
- **“Agents are unfairly evaluated because prompts didn’t induce verification loops”**: The paper provides qualitative trace evidence that one scaffold (Codex) did not self-check while Claude Code did iterate but still underperformed (Sec. 3, Fig. 8). This is a reasonable empirical finding about the tested scaffolds; stronger agent designs are a future-work suggestion rather than a current-paper flaw unless the paper overclaims generality (it mostly doesn’t beyond “agents … show no meaningful improvement,” which is scoped to the tested setups).

## Novel Insights
The paper’s own text essentially contains the key meta-critique of its evaluation: by explicitly admitting that strict rules trade off expressiveness and that instruction-following failures contaminate the intended construct (Sec. 2.4), it creates an internal inconsistency with the strength of the paper’s headline claims in the abstract/introduction. The most important improvement is therefore not adding more models, but aligning the benchmark’s public claim (“spatial intelligence”) with an evaluation that is demonstrably invariant to benign rendering variation and robust to room-identity permutations.

## Suggestions
- Redesign evaluation so **node correspondence is solved explicitly** (e.g., assignment/graph matching) rather than hard-coded by size rank, and report connectivity scores under that matching.
- Define baselines rigorously: include a **true random sampler** (matched on room count) and a **dataset-prior template** baseline, and avoid calling prior-shaped generation “random.”
- Document ground-truth adaptation: who did it, what rules for ambiguous cases, and inter-annotator agreement (even on a subset).

## Score and Decision
**Calibration anchors consulted (all retrieved):**
- High:  
  - /home/wg25r/review_agent/human_reviews_2026/DTQIjngDta.md (avg 8.0): strong method + well-validated evaluation; Blueprint-Bench is far weaker on metric validity.  
  - /home/wg25r/review_agent/human_reviews_2026/3vlMiJwo8b.md (avg 7.0): identifies benchmark shortcuts and responds with tighter benchmark design; Blueprint-Bench identifies its own evaluation pitfalls but does not resolve/validate them.
- Medium:  
  - /home/wg25r/review_agent/human_reviews_2026/dlaNQM6YbZ.md (avg 4.5): concerns about metric design sensitivity/interpretability; Blueprint-Bench shares similar evaluation-validity risk, but is more “benchmark proposal” than “benchmark analysis.”
- Low:  
  - /home/wg25r/review_agent/human_reviews_2026/TJWhvS5JXg.md (avg 1.2): withdrawn/very weak benchmark paper; Blueprint-Bench is substantially more complete/empirical than this, but still has core evaluation-validity gaps that affect acceptance.

**Overall assessment:** The research question is important and the pipeline is clearly described, but the central claims are not well supported because the metric is demonstrably misaligned (size-rank identity) and explicitly confounded with format/instruction-following, with no external validity check. Relative to the anchors, this lands below typical acceptable benchmark papers and closer to borderline/reject benchmark-validity cases.

MY FINAL SCORE: <pineapple>3.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>