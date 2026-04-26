Now I have enough information. Let me write the final review.

## Summary

This paper studies how to optimally allocate memory among model weights, KV cache, token budget, sampling group size, and KV cache compression for reasoning models under fixed memory constraints. Through 1,700+ experimental configurations on the Qwen3 family (0.6B–32B), with generalization experiments on DeepSeek-R1-Distill and OpenReasoning-Nemotron, the authors identify a scale-dependent "8-bit 4B" effective-size threshold below which memory should be allocated to model capacity rather than test-time compute, and above which the opposite strategy prevails. The paper also finds that the optimal weight precision, parallel scaling strategy, and KV cache compression method all depend on this effective-size threshold and task type.

## Strengths

- **Systematic and comprehensive empirical methodology.** The paper spans 1,700+ configurations across five axes of variation (model size, weight precision, token budget, group size, KV compression), addressing a multi-dimensional trade-off space that prior work has not tackled holistically. This breadth of measurement is genuinely impressive and provides practical value for deployers.

- **Central observation is well-supported and practically important.** The demonstration that KV cache can dominate total memory for reasoning models (e.g., 1.8× weights for Qwen3-4B at 32k tokens, Table 1) and that memory allocation strategies must shift with model scale is clearly illustrated through Pareto frontier plots. The contrast with Dettmers & Zettlemoyer (2023)'s 4-bit-optimal recommendation for non-reasoning models is sharp and well-motivated.

- **KV cache compression consistently advances the Pareto frontier (Finding 4).** This finding holds across all tested model sizes and precisions, providing a clear, robust, and actionable recommendation regardless of one's views on the threshold.

- **Clear, actionable findings and visualizations.** The five enumerated findings map to specific figures, and the Pareto frontier plots communicate multi-way trade-offs effectively. Practitioners can directly use these guidelines for deployment decisions.

- **Generalization beyond Qwen3.** The paper reproduces key findings (parallel scaling, scale-dependent thresholds) on DeepSeek-R1-Distill and OpenReasoning-Nemotron (Figures 6, 16), mitigating concerns that results are specific to one family.

## Weaknesses

### Fatal
None.

### Major

- **The "8-bit 4B" threshold is presented as a principled, universal dividing line but is derived from visual inspection of six discrete model sizes in one family.** The paper's central findings (Findings 1, 3, 5) all hinge on this threshold. However, it is identified by observing Pareto frontier transitions across {0.6B, 1.7B, 4B, 8B, 14B, 32B} in Qwen3, with no intermediate effective sizes tested in the critical 2.5–5.7 GB range (between 4B-4bit and 8B-4bit). No sensitivity analysis, uncertainty quantification, or statistical testing is provided for where the threshold lies. The generalization experiments on DeepSeek-R1-Distill and OpenReasoning-Nemotron test only a subset of configurations and are relegated to appendix figures without detailed discussion. The paper uses language like "effective size below 8-bit 4B" repeatedly as though it were a precise, transferable constant, when it may shift with model family, architecture, and task distribution. This matters because practitioners could make different deployment choices depending on whether the transition is sharp or gradual, and whether it sits at 3 GB, 4.2 GB, or 6 GB.

- **"Effective size" (parameters × bits/weight) conflates weight memory with model training quality, undermining the causal interpretation.** An 8B-4bit model (5.68 GB) and a 4B-8bit model (4.19 GB) occupy similar memory, but their representations were shaped by dramatically different training compute and data. When the paper finds that "memory is better spent on model weights" below the threshold, it may partly reflect that small models have too little training compute to benefit from extended reasoning, not that there is a fundamental memory-allocation principle. The paper never acknowledges this confound. The "where to spend bytes" framing is practical and valid for deployment decisions (you literally cannot deploy what doesn't fit in memory), but the causal claim that the threshold reflects an inherent property of "effective size" is not established.

### Minor

- **AIME25 floor effects may drive the task-dependent precision finding (Finding 2).** Finding 2 claims 4-bit is memory-optimal for "knowledge-intensive tasks" but not for mathematical reasoning, illustrated with GPQA-Diamond vs. AIME25. However, AIME25 is extremely difficult—many small/quantized configurations score near zero—which could create floor effects that make precision differences appear larger than they are. GPQA-Diamond allows more score gradation. The paper does not control for benchmark difficulty or report score distributions per configuration. This does not invalidate the qualitative finding that math reasoning is more precision-sensitive, but the confident "knowledge-intensive vs. mathematical reasoning" dichotomy may partly reflect benchmark psychometrics rather than a single clean task-type effect.

- **No variance or confidence interval reporting.** With 32 generations per instance, variance estimates are readily computable but are never reported. This makes it impossible to assess whether small differences in Pareto frontier position are meaningful, which is particularly important for pinning down the threshold claim.

- **Budget forcing may interact with model size/precision, which is not analyzed.** The forced "Wait" continuation is applied uniformly, but if it degrades output quality more for smaller/quantized models, this would disproportionately hurt smaller models on the Pareto frontier, potentially artificially sharpening the threshold. This is not investigated.

## Nice-to-Haves

- Testing intermediate model sizes or additional precision levels (e.g., 6-bit) near the proposed threshold would strengthen the threshold claim considerably.
- Reporting per-benchmark score distributions or difficulty-matched comparisons would strengthen the task-dependent precision finding.
- Explicit acknowledgment that "effective size" is a deployment-motivated proxy that conflates training quality with memory footprint, with discussion of whether the threshold reflects memory allocation or training compute.
- Prominently discussing the batched inference scenario (currently in Appendix C.3) in the main body, since deployment scenarios typically batch requests.

## Removed Points

- **"The PRM comparison tests only one large verifier."** The paper makes a specific empirical claim about ActPRM-X (13.28 GB) being memory-inefficient. This is a valid empirical observation, and the paper's language is appropriately specific. Removed because generalizing to "all external verifiers" would be a strawman—the paper doesn't claim that.

- **"R-KV vs. HQQ is a method-comparison, not strategy-comparison."** While true that this tests specific methods rather than abstract strategies, R-KV is specifically designed for reasoning models and HQQ is a standard general-purpose quantizer—this makes the comparison practically informative. Downgraded to a nice-to-have consideration rather than a weakness.

- **"The unbatched memory model limits practical applicability."** The paper addresses this in Appendix C.3, which analyzes the batched setting. The main text focuses on the single-request setting because it is the simplest and most common deployment scenario for reasoning models (where long generations dominate). This is a reasonable scoping decision, not a methodological flaw.

- **"The paper overstates the consensus on 4-bit optimality."** While the citation to Dettmers & Zettlemoyer (2023) is for a narrower set of tasks, the broader community does treat 4-bit as the go-to inference precision. This is a minor framing issue, not a substantive weakness.

- **"Missing related works."** Per review instructions, this is removed because I cannot confirm the existence of unspecified references.

- **Formatting and style complaints.** Removed per instructions.

## Novel Insights

The paper's most novel insight is that the memory allocation strategy for reasoning models is not just task-dependent but scale-dependent in a specific way: below a certain effective-size threshold, investing bytes in model capacity is strictly dominant, while above it, investing in test-time compute pays off. This reframes test-time scaling from a FLOPs-centric perspective (as in prior work) to a memory-centric one, which is the correct lens for deployment. The finding that KV cache compression universally advances the Pareto frontier—regardless of scale or precision—is a robust and immediately actionable contribution.

## Suggestions

- Soften the "8-bit 4B" threshold language from a universal constant to an empirical observation: "approximately 4 GB of weight memory in our tested configurations" or "we observe a transition near 8-bit 4B in the Qwen3 family."
- Add 95% confidence intervals (or at least standard error bars) to the key Pareto frontier figures, since threshold claims depend on whether small accuracy differences are statistically meaningful.
- Acknowledge the training-compute confound explicitly, e.g., "effective size groups together models with different training histories; we treat it as a deployment-level variable (what fits in memory) rather than a causal variable."

## Score and Decision

**Calibration comparison:**

- *High anchors*: Scaling Laws for Precision (avg 8.0, Oral) — rigorous theoretical framework validated across 465 runs; SpQR (avg 6.5) — focused near-lossless compression method with clear results.
- *Medium anchors*: Input-Adaptive Allocation (avg 6.5) — adaptively allocates computation, clear practical contribution; LLM-KICK benchmark (avg 6.75) — systematic empirical re-evaluation of compression; Inference Scaling Laws (avg 5.75) — similar empirical Pareto analysis of test-time compute.
- *Low anchors*: Balance Beam (avg 3.67) — poor theoretical analysis, limited scope; Scaling Laws for Mixed Quantization (avg 3.0) — intuitive claims, poor presentation; Universality from starvation (avg 3.0) — overclaimed universality from limited evidence.

This paper is substantially stronger than the low anchors (real systematic study, 1700+ configurations, clear practical recommendations) and comparable to medium anchors like LLM-KICK and Input-Adaptive Allocation. However, compared to the high-scoring Scaling Laws for Precision (avg 8.0), this paper lacks the theoretical grounding that makes that paper's conclusions robust—it derives its key threshold empirically from six model sizes in one family without sensitivity analysis. The overclaiming of the threshold as universal and the training-compute confound are meaningful weaknesses that push the score below the 7.0 tier but above the 5.0 borderline.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>