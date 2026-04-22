Now I have all the information needed. Let me compose the final review.

## Summary

Blueprint-Bench introduces a benchmark for evaluating spatial reasoning in AI by requiring models to convert apartment photographs into accurate 2D floor plans. The paper evaluates 8 systems across three architectures (LLMs, image generation models, and agents) on 50 apartments, reporting that most models perform at or below a "random baseline" while humans substantially outperform them, identifying a blind spot in spatial intelligence.

## Strengths

- **Novel and ecologically valid task design.** Converting real photographs into structured floor plans is an intuitive, grounded test of spatial understanding that—unlike many QA-based spatial benchmarks—requires models to produce a full spatial reconstruction rather than answer isolated questions. The ARC-inspired framing of testing out-of-distribution reasoning with in-distribution input (Section 1) is well-motivated.

- **Cross-architecture evaluation on a common numerical scale.** The paper evaluates LLMs (GPT-5, Claude 4 Opus, Gemini 2.5 Pro, Grok-4), image generation models (GPT-Image, NanoBanana), and agents (Codex CLI, Claude Code) on the same metric, enabling the first direct numerical comparison across these fundamentally different architectures (Figures 5, 7).

- **Key negative result: iterative agent refinement does not help.** The paper explicitly tests the hypothesis that humans outperform AI because they iteratively revise floor plans (Section 3). Figure 5 shows agent performance is not meaningfully better, and trace analysis (Figure 8) reveals *why*: Codex CLI never iterates, and Claude Code iterates but fails to correctly self-assess (claiming rooms are enclosed when they are not).

- **Deterministic, reproducible scoring algorithm.** Unlike LLM-as-judge approaches, the CV-based extraction and graph-comparison scoring (Section 2.3) is fully deterministic, making results reproducible. The 9 formatting rules (Section 2.1) ensure reliable extraction.

- **Honest and differentiated failure-mode analysis.** The paper identifies that NanoBanana fails due to instruction following (including furniture/windows, Figure 6), GPT-Image follows rules but lacks spatial intelligence, and agents fail for different reasons than single-pass models (Section 3). The acknowledgment that "Blueprint-Bench should test spatial intelligence, not instruction following" (Section 2.4) shows awareness of limitations.

## Weaknesses

### Fatal

None.

### Major

- **Size-rank-based room IDs create cascading penalties that undermine the precision of headline claims.** Rooms are assigned IDs by size rank (Section 2.3: "rooms are assigned unique IDs based on their size rank"), and the 50%-weight edge-overlap component uses these IDs. The paper acknowledges that "the penalty of making a mistake in the size ranking causes additional penalties when scoring the connectivity" (Section 2.4). This means a model with correct spatial connectivity understanding but incorrect size ranking is doubly penalized—the same error propagates into both the size and connectivity scores. The paper does not quantify how much of the low scores are attributable to this cascading effect vs. genuine spatial reasoning failure. Since the headline finding is that "most models perform at or below random," this unquantified artifact directly affects the interpretability of that claim. The paper's own observation that humans also suffered from this penalty ("they did not always get the size ranking correct...results in a harsh penalty") suggests the effect is non-trivial.

- **The metric conflates instruction-following failure with spatial intelligence failure, and no analysis disentangles them.** Models that cannot follow the 9 formatting rules (like NanoBanana and GPT-4o) score near zero for reasons unrelated to spatial reasoning. The paper acknowledges this conflation (Section 2.4: "Blueprint-Bench should test spatial intelligence, not instruction following") but treats it as an acceptable tradeoff rather than providing the analysis needed to support the central claim. A component-wise score breakdown—reporting performance on connectivity, size ranking, room count, etc. separately—would help distinguish models that fail on spatial reasoning (e.g., GPT-5, Gemini 2.5 Pro, which follow rules but still score low) from those that fail on formatting (e.g., NanoBanana). Without this decomposition, the claim that the benchmark reveals a "blind spot in spatial intelligence" is only partially supported for some models and entirely unsupported for others.

- **"Random baseline" is mislabeled—it is a strong prior baseline, not a chance-level baseline.** The paper describes generating this baseline as "generating typical floor plans using LLMs and image generation models without any image input" (Section 2.2), calling it a "worst-case baseline" there, but labels it "random baseline" in the abstract, figures, and results. Models with strong architectural priors about apartment layouts but no visual evidence do not constitute "random" performance. The phrase "most models perform at or below a random baseline" therefore misleadingly implies chance-level performance. Notably, the actual comparison—models with images failing to beat models without images—is arguably a *stronger* finding than merely failing to beat random chance, but the misleading terminology obscures this.

### Minor

- **Human–AI comparison uses different sample sizes and methodologies.** Human evaluation covers only 12 apartments vs. 50 for AI (Figure 7 caption). The human used an iterative approach while most AI models were single-pass. The paper tests the iteration hypothesis with agents and finds no improvement, but the agents' iterative behavior was itself flawed (Claude Code's "first generation was always much worse"), so the test is inconclusive. The gap direction (humans >> AI) is almost certainly real, but the magnitude is uncertain.

- **Composite scoring weights are unjustified.** The six components are combined with weights 50/20/10/10/5/5 (Section 2.3) with no sensitivity analysis or justification. The paper asserts that "two floor plans that follow the rules will always have a higher score if they are indeed similar" (Section 2.4) without proof. A sensitivity analysis showing the ranking of models is stable under alternative weight choices would strengthen confidence in the results.

- **No per-apartment difficulty analysis or error categorization.** The paper reports aggregated scores across all apartments but does not analyze which apartments are harder, whether failure modes differ by apartment complexity, or which specific spatial relationships (e.g., corridor connectivity vs. room adjacency) models struggle with most. This limits the diagnostic value of the benchmark.

### Trivial

- The dataset of 50 apartments is relatively small, though the paper keeps most data private to prevent overfitting, which is good practice.

## Nice-to-Haves

- A simplified task variant (e.g., providing room labels and sizes, asking only for connectivity) to isolate spatial reasoning from instruction-following and size-ranking, as this would directly validate the spatial intelligence claim.
- A room-label-based matching alternative (e.g., maximum-weight bipartite matching on room features) to eliminate the cascading penalty from size-rank IDs.
- Human evaluation on all 50 apartments with controlled methodology (single-pass) and inter-rater agreement, to make the human–AI comparison rigorous.
- A true random baseline (e.g., random graphs with realistic room-count distributions) to establish genuine chance-level performance for comparison.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Two floor plans with identical connectivity graphs but rooms in completely wrong geometric positions would score perfectly"** (Harsh Critic #1): While technically true that the metric does not capture room positions, this actually cuts *against* the critic's main argument—the metric is *more permissive* than full spatial matching, not less. If models still score poorly on this more permissive metric, it makes the finding *stronger*, not weaker. The paper already acknowledges it does not account for room shape (Section 2.4).

- **"The paper claims 'at or below random' but the baseline isn't random so the finding is uninterpretable"** (Harsh Critic #4, full version): The harsh critic argues this invalidates the finding entirely, but this overstates the case. If models with visual input cannot beat models without visual input, that is a meaningful finding—it just shouldn't be called "random." The terminology is misleading, but the comparison is informative.

- **"50 apartments is small for a benchmark intended to support numerical claims"** (Harsh Critic Section 2.1): This is a generic, one-size-fits-all criticism. Each apartment involves ~20 images and complex reasoning; 50 provides reasonable statistical power for the model-level comparisons shown. This is also standard for benchmarks of complex, multi-step tasks.

- **"No information about geographic diversity, apartment size range"** (Harsh Critic Section 2.1): This is a nice-to-have, not a weakness. The paper provides example apartments and describes the data collection process.

- **"Ground truth adapted from listing floor plans introduces potential noise"** (Harsh Critic Section 2.1): Speculative without evidence that this noise significantly affects the results.

- **"ARC analogy overstates the parallel because ARC's input is OOD"** (Harsh Critic Section 1): The paper explicitly makes this distinction: "We ask whether we can demonstrate such a blind spot using an input modality that is very much in distribution" (Section 1). The analogy is specifically about demonstrating blind spots, not about identical OOD structure.

- **"LLMs being poor at floor plan image understanding is a notable finding buried in limitations"** (Harsh Critic Section 2.4): This is a limitation of the extraction approach, not a new finding about model capabilities. The paper correctly frames it as a failed alternative.

- **Strength removed: "First direct numerical comparison of image generation models against their underlying LLMs"** — While valid as a contribution, the comparison is significantly confounded by instruction-following differences, limiting the insight.

## Novel Insights

The paper's finding that iterative agent refinement not only fails to help but fails for *architecturally distinct reasons*—Codex CLI doesn't iterate at all, while Claude Code iterates but cannot correctly self-assess its spatial output—represents a genuinely novel decomposition of agent failure modes. This suggests that the barrier to spatial reasoning in agents is not merely about providing more computational steps, but about the absence of reliable internal verification for spatial properties.

## Suggestions

- Add a component-wise score breakdown (connectivity, size ranking, room count, etc.) for each model to disentangle spatial reasoning failures from other failure modes.
- Relabel the "random baseline" as "zero-vision prior baseline" or similar, and recalculate a true random baseline (random graphs with realistic statistics) for comparison.
- Quantify the cascading penalty: report what happens to scores when room IDs are rematched using an optimal assignment (e.g., Hungarian algorithm) rather than size-rank-based IDs, to isolate connectivity understanding from size-ranking errors.

## Score and Decision

**Calibration anchors used:**

- **STARE** (avg 7.0, Accept Poster): `/home/wg25r/review_agent/human_reviews_2026/fbGmSV6tUw.md` — Spatial reasoning benchmark with 4K tasks, clean metrics (accuracy on QA), comprehensive human evaluation with timing. Blueprint-Bench is clearly below STARE: smaller scale (50 vs 4K), confounded metric, no timing data, less decomposable analysis.

- **SpaCE-10** (avg 6.0, Accept Poster): `/home/wg25r/review_agent/human_reviews_2026/Df7UjwEgIx.md` — Compositional spatial intelligence benchmark with 5K QA pairs, 811 scenes, 10 atomic + 8 compositional capabilities. Blueprint-Bench has more task novelty (generation vs. QA) but weaker metric validity and much smaller scale.

- **SpintBench** (avg 3.5, Reject): `/home/wg25r/review_agent/human_reviews_2026/IFNDogCGWV.md` — Spatial integration benchmark with limited scope (distance-only queries, 100 samples). Blueprint-Bench is clearly above SpintBench: more diverse model evaluation, more ecologically valid task, larger dataset.

- **MapQA** (avg 2.0, Reject): `/home/wg25r/review_agent/human_reviews_2026/dOISCbmkmG.md` — Map understanding benchmark with confounded evaluation (LLM-as-judge bias, COI). Blueprint-Bench has similar metric validity concerns but more transparent, deterministic scoring; no COI issues.

- **Beyond Benchmarks** (avg 4.0, Reject): `/home/wg25r/review_agent/human_reviews_2026/UJvub9fNws.md` — Identifies confounded metrics as a problem but has major methodological issues itself. Blueprint-Bench is comparable in that it identifies a real blind spot but has metric concerns.

- **AgentHard** (avg 3.33, Reject): `/home/wg25r/review_agent/human_reviews_2026/owFVvaLFdw.md` — Benchmark cleaning pipeline with circular validation. Blueprint-Bench has a cleaner methodology.

- **SpatialTree** (avg 4.0, Withdrawn/Reject): `/home/wg25r/review_agent/human_reviews_2026/EOTCSANb3l.md` — Hierarchical spatial benchmark with limited novelty.

Blueprint-Bench sits above the low-scoring papers (MapQA, SpintBench, AgentHard) because its task is more novel, its evaluation is more transparent, and it provides genuinely useful cross-architecture insights. It sits below SpaCE-10 (6.0) and STARE (7.0) because those have cleaner metrics, larger scale, and more decomposable analyses. The metric validity concerns (cascading size-rank penalty, instruction-following confound, mislabeled baseline) are real and affect the headline claims, but they don't invalidate the core observation—models genuinely struggle with this task. The paper is comparable to the medium-scoring spatial benchmarks with acknowledged limitations.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>