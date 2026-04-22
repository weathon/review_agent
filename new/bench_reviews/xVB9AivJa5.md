Now I have enough information to write the final review. Let me synthesize my assessment.

Key verified issues:
1. **Connectivity metric confounded with size ranking** — Real, verified from Section 2.4. The metric name "connectivity similarity" is misleading because errors in size ranking cascade into connectivity scoring. Major but acknowledged.

2. **"Random baseline" labeling** — Real, verified. The method section calls it "worst-case baseline" and describes it as LLM/image-generated typical floor plans, but in figures it's labeled "Random." This is a mislabeling issue that matters for the paper's central narrative claim.

3. **Instruction following confound** — Real, partially acknowledged by authors in Section 2.4. The paper recognizes this and analyzes Figure 6 to separate the two. Valid concern but partially addressed.

4. **Different data subsets for human comparison** — Real, verified from Figure 7 caption. However, both models AND humans in Figure 7 are on the same 12 apartments, so it IS apples-to-apples within that figure. The concern is more about generalizability from 12 apartments and not having human data on all 50. This is a minor concern.

5. **No metric validation** — The paper asserts the metric works but doesn't do perturbation experiments. Valid but more of a nice-to-have for a benchmark paper.

Let me now look at the calibration anchors carefully:

- Spatial reasoning benchmarks (STARE, SpatialViz-Bench) with good metrics got 6-7 scores
- Benchmarks with metric confounds undermining core claims got ~4.5-5 scores
- Truly flawed evaluation papers got 1.5-3 scores

Blueprint-Bench has a novel and interesting task design, good cross-architecture comparison framework, and clear qualitative analysis. But the metric has a significant confound (size ranking → connectivity), the "random" baseline is mislabeled, and the evaluation isn't validated beyond the assertion. These are meaningful but not fatal weaknesses — the core finding (models struggle with spatial reconstruction from photos) is likely directionally correct even if the specific numbers are noisy.

I'd put this around 4.5-5.5, reflecting that it's a worthwhile contribution with notable metric and evaluation weaknesses that partially undermine the strength of its quantitative claims.</think>Now I have all the information needed. Let me write the final consolidated review.

## Summary

Blueprint-Bench introduces a benchmark for evaluating spatial intelligence by requiring AI models (LLMs, image generation models, and agents) to convert apartment photographs into 2D floor plans, scored via a composite similarity metric based on room connectivity graphs and size rankings. The benchmark tests a compelling task—spatial reconstruction from in-distribution visual inputs—and finds that most models perform at or below a zero-image prior baseline while humans substantially outperform them, highlighting a genuine spatial intelligence blind spot.

## Strengths

- **Novel and well-motivated task design**: Unlike ARC-style benchmarks that use alien input modalities, Blueprint-Bench uses in-distribution inputs (photographs) paired with out-of-distribution reasoning requirements (spatial reconstruction), providing a complementary lens on AI capabilities. Section 1 clearly motivates this distinction: "While the input data is very much in distribution for how LLMs are trained, the task of translating it to a 2D floor plan is not something LLMs are trained for."

- **Cross-architecture numerical comparison on a unified task**: The benchmark evaluates LLMs, image generation models, and agents on the identical task with the same scoring metric, enabling the first direct numerical comparison across these architectures (Section 1: "To our knowledge, this is the first benchmark to make such comparisons"). Figure 5 delivers this comparison.

- **Transparent scoring decomposition**: The six-component similarity metric (Section 2.3: 50% edge overlap, 20% degree correlation, 10% density, 10% room count, 5% door count, 5% door orientation) is explicitly specified, and the authors honestly discuss limitations in Section 2.4, including rejected alternative approaches with empirical justification (LLM extraction failed, point sampling "harshly penalized small mistakes in unpredictable ways").

- **Insightful qualitative analysis of failure modes**: Figure 6 disentangles instruction-following failures (NanoBanana adding furniture, GPT-4o missing dots) from genuine spatial reasoning failures (GPT Image following rules but scoring at baseline), and Figure 8's agent trace analysis provides mechanistic insight into why iterative refinement fails (Codex doesn't iterate; Claude Code iterates but can't detect its own errors).

- **Clear core finding with broad interest**: The result that even top models barely outperform a zero-image prior baseline when reconstructing spatial layouts from photographs documents a genuine and important capability gap.

## Weaknesses

### Fatal
None.

### Major

- **The "connectivity similarity" metric is confounded with size-ranking accuracy, undermining interpretability of the headline metric**: The scoring algorithm assigns room IDs by size rank, then computes edge overlap between generated and ground-truth graphs (Section 2.3). If a model correctly identifies all room adjacencies but misorders two rooms by size, those rooms receive different IDs, causing the edge overlap component (50% of the total score) to penalize the model despite correct spatial reasoning. The authors acknowledge this in Section 2.4 ("the penalty of making a mistake in the size ranking causes additional penalties when scoring the connectivity") but frame it as a known tradeoff rather than recognizing that it makes the named metric — "connectivity similarity" — not purely measuring connectivity. A low score cannot be attributed to poor spatial reasoning versus poor size estimation, which weakens the paper's central claims about AI spatial intelligence. This could be addressed with a maximum-weight graph matching alignment instead of size-rank alignment.

- **The "random baseline" is a prior-only baseline, not a random baseline, and this mislabeling changes the paper's core narrative**: Section 2.2 describes generating "typical floor plans using LLMs and image generation models without any image input"—a zero-image-prior baseline. Yet in Figures 5 and 7 and throughout Section 3, this is labeled "Random." LLM-generated "typical" floor plans encode strong priors about apartment topology (kitchens near living rooms, bathrooms are small), so a score of ~0.28–0.32 is not what one would get from random graphs. The abstract's claim that models "perform at or below a random baseline" is therefore misleading; the accurate statement is that models barely improve over their priors when given actual apartment images. This is still a meaningful finding, but substantively different from the "at or below random" framing.

### Minor

- **The benchmark structurally conflates instruction following with spatial intelligence, though the authors partially address this**: Section 2.4 states "Blueprint-Bench should test spatial intelligence, not instruction following," yet models that violate the 9 formatting rules (GPT-4o, NanoBanana) receive very low scores (~0.15–0.18) that cannot distinguish poor spatial reasoning from poor rule compliance. The authors partially address this via qualitative analysis in Figure 6, showing that some models fail on following rules while others fail on spatial reasoning. Still, the aggregate scores for instruction-following violators are included in the comparison and contribute to the "models perform poorly" narrative without clear decomposition.

- **Human evaluation on only 12 of 50 apartments limits generalizability of the human-AI gap**: Figure 7 is based on a 12-apartment subset, and the human baseline (0.547) is only available for this subset. While model scores in Figure 7 are computed on the same 12 apartments (making the within-figure comparison valid), the human-AI gap may not generalize, particularly since no information is provided about how these 12 apartments were selected.

- **The weighting of the composite metric is not justified or sensitivity-tested**: The 50/20/10/10/5/5 weighting in Section 2.3 is presented without justification. Whether the central claim—that models struggle with spatial intelligence—is robust to alternative weightings is unknown, though the core finding (models near baseline) is likely not sensitive to this choice.

- **Claim that agent-based iterative refinement shows "no meaningful improvement" is based on only two agents with different scaffolds**: Codex (GPT-6) didn't use the iteration capability, while Claude Code did iterate but from a weaker starting point. This tests specific agent implementations, not the concept of iterative refinement itself.

### Trivial
None worth listing beyond what's already noted.

## Nice-to-Haves

- **Maximum-weight graph matching for room alignment**: Replacing size-rank alignment with a matching algorithm that maximizes edge overlap would decouple connectivity scoring from size estimation, making the metric genuinely measure what it claims to measure.
- **True random baseline**: Generating random room graphs (random room count, random connectivity) would establish a genuine floor for comparison and clarify how much of the prior-only baseline's score comes from structural priors.
- **Metric validation via controlled perturbations**: Invert two room sizes, remove one edge, add one room in ground-truth plans and verify that the metric's output reflects expected deltas.
- **Human evaluation on the full 50-apartment set** to establish a more robust estimate of the human-AI gap.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Most models labeled as 'Image model' in the results table"**: This appears to be a parser extraction artifact from the figure/table. The original Figure 5 uses visual differentiation (striped bars for image models, dotted for agents), and the text in Section 2.2 correctly distinguishes LLMs from image models. This is not a substantive author error.

- **"No statistical significance tests despite claiming models 'statistically perform better'"**: This is a reproducibility/process nitpick. The paper reports means with error bars (standard deviation) across 50 apartments and multiple epochs, which provides substantial empirical evidence for the claims, even without formal hypothesis testing.

- **"Missing related work survey on cross-architecture benchmarks"**: Claiming the paper should cite specific related works without having external sources to confirm their existence would be making things up.

- **"50 apartments is a small dataset"**: The benchmark is designed as a difficult, carefully curated evaluation suite, not a training set. 50 apartments with ~20 images each provides 50 independent evaluation points, which is standard for benchmark papers of this type (e.g., ARC uses a similar scale).

- **"Agent setup insufficient for reproducibility"**: This is a minor implementation detail concern. The paper describes the Docker setup, image access, and tool use. The code is open-sourced.

- **"Ground-truth floor plans come from marketing materials"**: The paper clearly states these are adapted from apartment listing floor plans, which are the natural ground truth for this task. Criticizing the choice of ground truth is scope creep for a benchmark evaluating model reconstruction capability.

## Novel Insights

The separation between instruction-following failures and spatial-reasoning failures is genuinely diagnostic. The paper shows that some models (GPT-4o, NanoBanana) fail primarily on rule compliance, while others (GPT Image) comply with rules but still produce spatially inaccurate plans—these are distinct failure modes requiring different interventions. The agent trace analysis further reveals that iterative refinement doesn't help not because iteration is inherently useless, but because current agents cannot reliably detect their own spatial errors (Claude Code confidently asserts "Each room is fully enclosed" when it isn't). This suggests the bottleneck is self-evaluation, not iteration per se.

## Suggestions

- Rename the "random baseline" to "zero-image prior baseline" throughout the paper, and add a true random graph baseline. This small change would make the narrative more precise: "models barely improve over their priors" rather than "models perform at random."
- Add a decomposition analysis: report size-ranking accuracy separately from connectivity accuracy, so readers can interpret what drives low scores. This would turn the confound from a weakness into a diagnostic feature.
- Consider maximum-weight bipartite matching for room alignment instead of size-rank alignment, which would make the 50% edge-overlap component genuinely measure connectivity.

## Assessment

**Originality**: The task of converting apartment photos to floor plans is a creative and well-chosen test of spatial intelligence. The cross-architecture comparison framework (LLMs, image models, agents on the same task) is a genuine contribution. The idea of using in-distribution inputs with out-of-distribution reasoning demands is novel.

**Importance**: The spatial intelligence blind spot is a real and important finding. The framework provides a useful tool for tracking progress on this capability.

**Claim support**: The core qualitative finding (models struggle, humans excel) is well-supported. The quantitative claims are weakened by the metric confound and baseline mislabeling. The absolute scores should be interpreted with caution.

**Soundness of experiments**: The evaluation is systematic across 8+ models, but metric validation and baseline correctness are concerns.

**Clarity**: The writing is clear and well-structured. The 9 formatting rules are precisely defined. Figures are informative.

**Value to community**: The benchmark addresses a genuine gap. Even with metric issues, it serves as a useful starting point for tracking spatial intelligence progress.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| STARE (fbGmSV6tUw) — spatial reasoning MLLM benchmark | 7.0 | Blueprint-Bench has a similar goal but less rigorous metric validation; STARE has cleaner evaluation |
| SpatialViz-Bench (OqZ7bm28Xx) — spatial visualization benchmark for MLLMs | 6.0 | Comparable topic; this paper has similar scope but Blueprint-Bench has more significant metric confounds |
| LEGO-Puzzles (jQh9SUrnev) — multi-step spatial reasoning benchmark | 5.5 | Similar "models fail at spatial" finding; LEGO-Puzzles rejected despite less severe metric issues |
| MMMG (Eo2OSOQL1P) — multimodal generation benchmark with unreliable metrics | 5.5 | Blueprint-Bench shares the "metric doesn't fully validate what it claims" concern |
| DNA shuffling benchmark (ph9Pq45KLN) — fundamentally flawed evaluation | 1.5 | Much more severe metric flaw (hardware dependency makes scores non-reproducible); Blueprint-Bench's issues are less fundamental |
| Benchmark confounds papers (g03rPJwRwS, xkbjNJi0eb, UJvub9VBr) — confounds undermining claims | 4.0-4.5 | These papers have core metric confounds similar in severity to Blueprint-Bench's size-ranking↔connectivity confound |

Blueprint-Bench sits in a similar space to the 4.5–5.5 range: the contribution is real and interesting, but the metric confound and baseline mislabeling are substantive enough to weaken the quantitative claims. It's above the truly flawed benchmark papers (1.5–3.0 range) because the core finding is directionally correct and the task design is valuable, but below well-validated spatial benchmarks (6.0–7.0 range) because the metric doesn't cleanly measure what it claims.

## Score and Decision

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>