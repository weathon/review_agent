=== CALIBRATION EXAMPLE 19 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the paper's contribution. The abstract is honest about limitations and results. However, several concrete claims require scrutiny:

- The abstract states "most models perform at or below a random baseline" — yet the random baseline itself is constructed by running LLMs/image models without image input (Section 2.2), which is not a conventional random baseline. The relationship between this "no-image" baseline and a theoretically grounded random floor-plan graph is never established. A model performing "at" this baseline could actually be doing something useful if the baseline is poorly calibrated.
- The abstract claims Blueprint-Bench provides "the first numerical framework for comparing spatial intelligence across different model architectures." While plausible, existing spatial benchmarks (e.g., SpatialBench, VSR, or embodied navigation benchmarks) are not discussed, leaving the novelty claim unsubstantiated.

---

### Introduction & Motivation

The motivation is intuitively compelling — can general-purpose models demonstrate spatial intelligence without spatial-specific training? The analogy to ARC is apt. However:

- **ARC analogy overstated.** ARC is hard for LLMs precisely because the input *and* output modality are out-of-distribution. Blueprint-Bench inputs (apartment photos) are in-distribution; the authors claim the *output* (floor plan) is the challenge. But LLMs routinely generate SVG code, and floor plan SVGs plausibly appear in pre-training data. The paper does not address whether models may have been trained on floor plan generation tasks, making the "out-of-distribution" claim uncertain.
- **Image model motivation is thin.** The argument that "image generation models lack numerical benchmarks" motivates part of the work, but the chosen task (floor plan reconstruction) is not naturally aligned with what image generation models are designed for. A model like NanoBanana or GPT-Image is optimized for perceptual quality generation, not structured spatial reasoning. Conflating poor task performance with lack of general intelligence is a significant conceptual leap.
- **Contributions are not explicitly enumerated.** ICLR reviewers expect a clear bulleted list of contributions; the introduction lists them only implicitly.

---

### Method — Dataset (Section 2.1)

**Dataset size is critically small.** Only 50 apartments are used for the main results, and the human comparison (Figure 7) is limited to just 12. For a benchmark paper, 50 data points is insufficient for reliable ranking of models — particularly when error bars in Figure 5 appear large. The variance within models across apartments is uncharacterized.

**Ground truth construction is underspecified.** The floor plans are "adapted from the apartment listing's official floor plan image," but:
- Who performed this adaptation? Was it a single annotator or multiple?
- Is inter-annotator consistency measured?
- How much manual curation went into enforcing the 9 rules on the ground truth?
- Apartment listings vary enormously in floor plan accuracy; the fidelity of the ground truth to the actual apartment layout is assumed but not validated.

**The 9 formatting rules create a confound.** As the authors acknowledge in Section 2.4, the benchmark conflates *spatial intelligence* with *instruction following*. GPT-4o and NanoBanana are penalized not for spatial reasoning failures but for formatting non-compliance. The paper does not disentangle these. A model that perfectly understands the spatial layout but produces a slightly non-compliant rendering will score poorly, while a model that produces a compliant but spatially wrong plan will score better. This fundamentally undermines the paper's core claim of measuring spatial intelligence.

---

### Method — Evaluation / Scoring (Section 2.3)

This is the weakest section of the paper and represents a serious methodological concern.

**Arbitrary weight choices.** The composite score uses weights (50% edge overlap, 20% degree correlation, 10% density, 10% room count, 5% door count, 5% door orientation) with no principled justification, no ablation, and no sensitivity analysis. The conclusions of the entire paper are contingent on these weights. A different weighting could change model rankings.

**Room identification by size rank is problematic.** Rooms are assigned IDs based on their area rank (1 = largest). This means:
1. If two rooms have similar sizes, a minor measurement error changes which room is "room 1" vs "room 2," causing cascading errors in the connectivity graph comparison.
2. Apartments with many similar-sized rooms will be harshly penalized for ordering errors that have nothing to do with connectivity understanding.
The authors acknowledge this in Section 2.4 but present it as a minor limitation. It is actually central: it means the edge overlap metric (50% of the score) is entangled with a brittle room-ordering heuristic.

**Score not validated against human judgment.** There is no experiment showing that the composite score correlates with human assessments of floor-plan similarity. This is a fundamental gap for a benchmark paper — without this validation, it is unclear whether a score of 0.3 vs 0.4 means anything semantically meaningful.

**Random baseline methodology.** Generating floor plans "without image input" using LLMs is not a proper random baseline. The resulting baseline inherits whatever prior LLMs have about typical floor plan layouts. A proper random baseline would sample random connectivity graphs from a distribution matching the statistics of the ground truth graphs (e.g., random spanning trees with n rooms). The current baseline is under-defined and potentially inflated or deflated relative to chance.

**Door orientation metric.** Including door orientation (horizontal vs. vertical) as 5% of the score is problematic — orientation is often arbitrary (e.g., a hallway could run either way) and penalizes correct connectivity predicted in a mirrored layout.

---

### Method — Generation (Section 2.2)

**Heterogeneous prompting across model types.** LLMs are asked to generate SVG code while image generation models produce raster images directly. This is not an apples-to-apples comparison — SVG generation involves a coding step that favors models with strong code generation, while raster generation tests a different capability entirely. The paper compares scores across these paradigms as if they measured the same thing.

**Agent comparison confounds base model with scaffold.** Codex CLI presumably uses GPT-5, and Claude Code uses Claude 4 Opus. These are different base models. Any performance difference between agents could be entirely attributable to base model differences rather than the agentic loop. A clean ablation would compare (a) GPT-5 direct, (b) GPT-5 via Codex, (c) Claude 4 Opus direct, and (d) Claude 4 Opus via Claude Code.

**Number of inference runs / epochs not specified in the main text.** The caption for Figure 5 mentions "aggregated results... across apartments and epochs," but the number of epochs (independent runs per apartment) is never stated in the methods section. This is critical for understanding variance estimates.

---

### Results & Discussion (Section 3)

**Statistical significance claims are vague.** The text states that "GPT-5, Gemini 2.5 Pro, GPT-5-mini, and Grok-4 statistically perform better than the random baseline," but no statistical test, p-value, effect size, or multiple-comparison correction is reported anywhere. With 50 apartments and high within-model variance (evidenced by large error bars), this claim needs substantiation.

**Figure 7 inconsistency.** Human performance is evaluated on only 12 of the 50 apartments, while model performance is evaluated on all 50. These two populations are not directly comparable. A single human performing the task is also an n=1 measurement — there is no variance in human performance reported, which is suspicious since humans also vary in spatial reasoning ability.

**Agent analysis is primarily anecdotal.** The finding that "Claude Code iterates but still doesn't improve" is based on qualitative trace inspection, not systematic analysis. How many traces were inspected? What fraction of runs showed iterative refinement behavior? What was the distribution of revision counts? This analysis needs quantification.

**The observation that agents fare no better than direct models is potentially the most interesting finding.** Yet the paper spends more time on the trivial finding that models underperform humans. Why does iterative refinement not help? The paper says "the reasons... require further investigation" — but this is the authors' own paper. More investigation should have been done before submission.

**Figure 4 reference error.** The text says "see Figure 4 for an example" of GPT Image's compliant output, but Figure 4 is described in the caption as showing the extraction algorithm output. This seems like a mis-reference.

---

### Writing & Clarity

The paper is generally readable but several passages are confusing or incomplete enough to affect understanding:

- The appendix (Section A) is entirely blank — "Appendix" and two blank pages. The main text promises "detailed graphs with results per data point can be found in Appendix" (Section 3), but nothing is there. This is a significant omission.
- The relationship between "epochs" (mentioned in Figure 5 caption) and experimental runs is never defined.
- Section 2.2 begins mid-sentence: "The third model-type we test on Blueprint-Bench In addition to LLMs and image generation models..." — a clear editing error.

---

### Limitations & Broader Impact

**Section 2.4** is unusual for a paper in that it appears within the methods section rather than as a standalone section. The limitations acknowledged are real, but several important ones are absent:

1. **No discussion of data leakage.** Apartment floor plans are commonly included in real estate websites that form part of web-crawled pre-training corpora. Some models may have seen floor plans corresponding or similar to the evaluated apartments.
2. **The benchmark cannot distinguish spatial understanding from spatial rendering.** A model that mentally constructs the correct layout but fails to express it in SVG/pixel format would score poorly, not because of spatial failure but output format failure.
3. **Single-task benchmark.** Floor plan generation is one very specific spatial task. Generalizing from performance here to "spatial intelligence" broadly is a stretch — models that fail here might excel at object affordance, navigation, or 3D reasoning.
4. **The ethics statement is perfunctory.** The connection between floor plan understanding and "military robotics" is asserted without analysis.

---

### Overall Assessment

Blueprint-Bench is a creative and practically motivated benchmark addressing a genuine gap — the absence of structured spatial reasoning evaluations for general-purpose multimodal models. The task design (photo → floor plan) is clever in that it uses in-distribution inputs to probe out-of-distribution output capabilities. However, in its current form the paper has several serious methodological weaknesses that undermine the credibility of its conclusions.

The most critical issues are: (1) the scoring metric relies on size-rank-based room identification, an inherently brittle heuristic that the authors themselves flag but do not resolve; (2) the composite score's arbitrary weights are never ablated, making all quantitative comparisons fragile; (3) the random baseline is not a true random baseline; (4) the blank appendix contradicts the main text's promise of detailed per-apartment results; (5) the agent analysis conflates base model identity with scaffold design; and (6) the benchmark's core claim of measuring "spatial intelligence" is confounded with instruction-following ability. The dataset (50 apartments, 1 human annotator, 12-apartment human comparison) is too small for a benchmark paper making ranking claims. The writing is occasionally sloppy. While the research direction is worth pursuing and the finding that iterative agents do not help is genuinely interesting, the paper needs substantially more methodological rigor, a properly filled appendix, validated scoring metrics, and a cleaner experimental design before it meets ICLR's standard.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Blueprint-Bench, a new benchmark designed to evaluate the spatial reasoning capabilities of Large Language Models (LLMs), Image Generation models, and AI agents. The task requires converting a set of interior apartment photographs into a standardized 2D floor plan, which is then quantitatively scored using an automated algorithm based on room connectivity graphs and size rankings. The results indicate a significant performance gap between current AI systems and humans, highlighting a "blind spot" in spatial intelligence despite the input modality being within the models' typical training distribution.

### Strengths
1.  **Comprehensive Multimodal Comparison:** The benchmark uniquely compares diverse model families (LLMs, Image Gen, Agents) on the same spatial reasoning task. This directly addresses the authors' claim in the Introduction regarding the lack of numerical evidence for "general intelligence" in image generation models (Section 1).
2.  **Robust Automated Scoring Mechanism:** The evaluation pipeline is well-defined, utilizing a computer vision-based extraction step to convert images into structured JSON connectivity graphs and size rankings (Section 2.3). This provides a clear, objective metric compared to human preference studies often seen in image generation literature.
3.  **Inclusion of Human Baseline:** By including human performance as a comparison metric (Figure 7), the paper grounds the AI results in reality, reinforcing the argument that current AI systems are missing fundamental capabilities. The human approach (iterative drawing) is also analyzed to contextualize AI failures (Section 3).

### Weaknesses
1.  **Conflation of Instruction Following and Spatial Intelligence:** The authors acknowledge that strict visual rules (e.g., "Lines are straight... 3 pixels wide", "Pure red, pure black..." colors) are used to enable scoring, but admit this tests instruction following as much as spatial ability (Section 2.4). Models with poor instruction following (e.g., NanoBanana) may be unfairly penalized on spatial logic, making the "Spatial Intelligence" claim ambiguous.
2.  **Baseline Definition and Dataset Scale:** The dataset consists of only 50 apartments, which is small for a rigorous benchmark of general intelligence. Furthermore, the "random baseline" in Figure 5 is described as "generating typical floor plans... without any image input" (Section 2.2), which is semantically a "hallucination baseline" rather than a true random geometric baseline. The motivation for the specific weights (50% edge overlap, 20% degree correlation, etc.) in the scoring algorithm also lacks empirical justification (Section 2.3).
3.  **Limited Failure Mode Analysis:** While the paper notes models perform "at or below random baseline," the analysis of *why* the models failed is superficial. It attributes most errors to instruction following (e.g., including furniture) rather than analyzing spatial reasoning failures (e.g., incorrect connectivity or topology) in depth. There is no ablation study to determine whether the failure lies in the vision encoder, the reasoning capability, or the image decoding.

### Novelty & Significance
**Novelty:** The benchmark itself is novel in its specific construction of floor plans from photos for evaluation purposes, but the broader task has precedents (e.g., NeRF, LayoutGPT). The specific contribution of comparing Image Generation models to LLMs under this unified spatial constraint is more novel. The scoring methodology (graph-based similarity) is technically appropriate but standard graph matching rather than a significant methodological advancement.

**Significance:** The results are significant for the community as they challenge the narrative that current generalist models possess robust spatial reasoning. However, the significance is tempered by the dataset size and the scoring rigidity. If adopted, this benchmark could drive research into how LLMs represent physical space, but currently, it serves more as a diagnostic than a driver of architectural innovation without deeper causal analysis.

### Suggestions for Improvement
1.  **Decouple Instructions from Scoring:** Consider revising the evaluation strategy to be less reliant on strict pixel-perfect formatting. For example, allow models to output raw floor plans and use a more robust geometric matching algorithm that tolerates stylistic variations, ensuring a purer measure of spatial reasoning. Alternatively, split the evaluation into an "Instruction Adherence" score and a "Spatial Accuracy" score.
2.  **Expand and Justify Dataset:** Increase the number of apartments to improve generalizability across different architectural styles. Additionally, provide a clear rationale or sensitivity analysis for the scoring weights (e.g., why edge overlap is weighted at 50% vs. 20% for degree correlation).
3.  **Deepen Failure Analysis:** Conduct a more rigorous error analysis to distinguish between visual hallucination (seeing a wall that doesn't exist) and spatial reasoning errors (incorrect topology). An ablation study removing the language component (if possible) or vision component would clarify the bottleneck.
4.  **Clarify Baselines:** Explicitly define the "random baseline" metric. A true random graph generator should be used to establish the floor for what a model without any knowledge could achieve, rather than a "typical hallucination" baseline which implies some prior knowledge.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Specialized Reconstruction Baselines:** Add comparisons against dedicated SLAM or floor-plan reconstruction models (e.g., CubeNet, LayoutNet) to establish an upper performance bound. Without this, it is unclear if the task is impossible for *all* AI or merely generalist models, undermining the "spatial intelligence blind spot" claim.
2. **Consistent Human Baseline:** Evaluate human performance on the full 50-apartment set rather than a subset of 12. Comparing model scores on 50 samples against human scores on 12 invalidates the statistical significance of the claimed performance gap.
3. **Input Protocol Ablation:** Specify and ablate how the ~20 images are fed to models (e.g., concatenation vs. summarization vs. sliding window). Performance may be limited by context window constraints rather than spatial reasoning, which this experiment would reveal.
4. **Instruction-Following Control:** Include a control task that requires strict formatting without spatial reasoning to decouple instruction-following failures from spatial reasoning failures. Currently, models failing to draw red dots are scored as 0, conflating two distinct capabilities.

### Deeper Analysis Needed (top 3-5 only)
1. **Semantic Validity of Metric:** Analyze scores where room types are swapped (e.g., kitchen vs. bedroom) but topology remains correct. The current metric ignores semantics, meaning a functionally useless floor plan can receive a high score, undermining the "accurate reconstruction" claim.
2. **Error Component Breakdown:** Provide a table breaking down scores by the six similarity components (connectivity, size, density, etc.). Aggregated scores hide whether models fail due to geometry, counting, or graph structure, preventing diagnostic insight.
3. **Random Baseline Definition:** Explicitly define the generative process for the "random baseline." Without knowing how random graphs are generated, claims that models perform "at or below random" are unverifiable and potentially misleading.
4. **Context Window Correlation:** Correlate model performance with context window size and image resolution. If performance drops as image count increases, the bottleneck is memory/context, not spatial intelligence.

### Visualizations & Case Studies
1. **High-Score Failure Cases:** Display generated floor plans that receive high similarity scores but are semantically incorrect (e.g., missing rooms, swapped functions). This exposes whether the scoring algorithm aligns with human judgment of quality.
2. **Agent Refinement Trajectories:** Plot the similarity score at each iteration for agent-based approaches. This reveals whether refinement moves toward the solution or oscillates, validating the claim that iterative methods offer no benefit.
3. **Score Distribution Histograms:** Replace mean bar charts with distribution histograms for models and baselines. Means obscure variance; overlapping distributions would weaken the claim that models are statistically distinct from random.

### Obvious Next Steps
1. **Release Full Test Set to Reviewers:** ICLR reproducibility standards require reviewers to verify results. Keeping the majority of the dataset private prevents validation of the claims and risks rejection.
2. **Standardize Input Preprocessing:** Publish the exact code used to resize, order, and prompt models with the 20 images. Without this, other researchers cannot replicate the experimental conditions or compare new models fairly.
3. **Integrate Semantic Room Labeling:** Update the scoring algorithm to verify room types (kitchen, bath, etc.) via OCR or metadata. A floor plan benchmark must evaluate functional accuracy, not just topological connectivity.
4. **Define Model Versions and Dates:** Specify exact model version IDs and inference dates. "GPT-5" and "Claude 4" are insufficient without specific build numbers, as capabilities shift rapidly with updates.

# Final Consolidated Review
## Summary
Blueprint-Bench introduces a benchmark for evaluating spatial reasoning in AI models through the task of converting apartment photographs into 2D floor plans. The paper compares LLMs (GPT-5, Claude 4 Opus, Gemini 2.5 Pro, Grok-4), image generation models (GPT-Image, NanoBanana), and agent systems against human and random baselines, finding that most models perform at or below random while humans substantially outperform them.

## Strengths
- **Novel cross-architecture comparison:** The paper provides the first systematic numerical comparison of spatial reasoning capabilities across fundamentally different model families—LLMs, image generation models, and agents—on the same task. This addresses a genuine gap noted by the authors: image generation models lack numerical benchmarks comparable to those standard for LLMs.
- **Clear task design for probing spatial intelligence:** The photo-to-floor-plan task cleverly uses in-distribution inputs (photographs) to probe an out-of-distribution capability (spatial reconstruction), analogous in spirit to ARC but with more naturalistic stimuli.
- **Human baseline provides meaningful grounding:** By including human performance, the paper establishes that the task is fundamentally solvable and quantifies the capability gap. The observation that humans correctly identified room connectivity (though not always size rankings) offers interpretive context for model failures.
- **Interesting negative finding about agents:** The result that agentic systems with iterative refinement show no improvement over single-pass generation is counterintuitive and meaningful—agents had the same affordances as humans (multiple views, iterative editing) yet failed to leverage them.

## Weaknesses
- **Scoring metric relies on arbitrary weights without justification or ablation.** The composite score uses weights (50% edge overlap, 20% degree correlation, 10% density, 10% room count, 5% door count, 5% door orientation) with no principled derivation. Different weightings could change model rankings, and no sensitivity analysis is provided. This is critical because all quantitative conclusions depend on these choices.

- **Size-rank room identification creates cascading errors.** Rooms are identified by area ranking (largest = room 1), meaning a small measurement error in similar-sized rooms can swap identities and invalidate connectivity scoring. The authors acknowledge this in Section 2.4 but treat it as minor—it actually undermines the primary metric (50% of the score).

- **"Random baseline" is not properly defined.** The baseline is described as "generating typical floor plans using LLMs and image generation models without any image input" (Section 2.2). This is a hallucination baseline that inherits LLM priors about typical layouts, not a random graph baseline. Claims that models perform "at or below random" are therefore misleading without defining what random means.

- **Human evaluation uses different dataset than model evaluation.** Figure 7 compares human performance on 12 apartments against model performance on 50 apartments. These are different populations, making statistical comparison invalid. No variance is reported for human performance (apparently n=1), further undermining the comparison.

- **No validation that scores correlate with human judgment of similarity.** For a benchmark paper, it is essential to show that the automated scoring reflects human perception of floor-plan correctness. Without this, we cannot interpret whether a score of 0.3 vs 0.4 is meaningfully different.

- **Benchmark conflates instruction-following with spatial reasoning.** Models like NanoBanana that fail to follow the 9 formatting rules (e.g., including furniture, wrong colors) receive low scores regardless of their spatial understanding. The authors acknowledge this but do not attempt to disentangle the two capabilities.

- **Agent comparison confounds base model with scaffold.** Codex CLI uses GPT-5 while Claude Code uses Claude 4 Opus—different base models. Any performance difference cannot be attributed to the agent scaffold versus the underlying model capability.

- **No statistical significance tests reported.** The paper claims certain models "statistically perform better than the random baseline" but provides no p-values, confidence intervals, or multiple-comparison corrections. Given the large error bars in Figure 5, this claim needs substantiation.

- **Ground truth construction is underspecified.** The paper states floor plans are "adapted from the apartment listing's official floor plan" but does not specify who performed adaptations, whether inter-annotator agreement was measured, or how faithfully the adapted plans match actual apartments.

- **Dataset size is small for a benchmark.** Fifty apartments is limited for reliable model ranking, particularly given the apparent high variance across apartments.

- **Appendix referenced in main text is blank.** The paper promises "detailed graphs with results per data point can be found in Appendix" but the provided appendix contains no content, preventing verification of per-apartment results.

## Nice-to-Haves
- **Error component breakdown:** A table showing performance on each scoring component (connectivity, room count, door count, etc.) would help diagnose whether models fail on graph structure, counting, or geometry.
- **Comparison with specialized reconstruction models:** While outside the paper's scope of testing generalist models, including one specialized floor-plan reconstruction baseline would clarify the task's difficulty ceiling.
- **Input protocol specification:** The paper should specify how ~20 images are presented to models (concatenated, sequential, with context window handling) as this could be a bottleneck separate from spatial reasoning.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"First numerical framework" novelty dispute (Harsh Critic):** The claim is reasonable and the comparison to existing spatial benchmarks (VSR, SpatialBench) is speculative—the critic does not establish these address the same cross-architecture comparison goal.

- **ARC analogy overstated (Harsh Critic):** The analogy is apt. Whether LLMs have seen floor plan SVGs in training is speculative; the paper's point that the task requires genuine spatial reasoning stands.

- **Perfunctory ethics statement (Harsh Critic):** Standard length for ICLR papers. The military robotics connection is a reasonable, if brief, observation about dual-use potential.

- **Release full test set (Spark Finder):** Keeping test data private to prevent overfitting is standard benchmark practice. Requiring public release conflicts with benchmark integrity.

- **Specialized SLAM/LayoutNet baselines (Spark Finder):** The paper explicitly scopes to evaluating generalist models, not comparing against specialized systems. This is scope creep.

- **Semantic room labeling requirement (Spark Finder):** The paper discusses this limitation. Room types are intentionally not labeled to avoid introducing prior assumptions about which rooms should be largest—a valid design choice.

## Novel Insights
The most striking finding is that agentic scaffolding provides no benefit despite offering the same affordances humans use (iterative viewing and editing). The qualitative trace analysis showing Claude Code iterates but fails to improve, while Codex doesn't even attempt iteration, suggests current models lack the meta-cognitive awareness to effectively exploit multi-step workflows for spatial tasks. This challenges the assumption that agentic architectures automatically unlock latent capabilities—models may need fundamentally different training to benefit from iterative refinement.

## Suggestions
- **Validate the scoring metric:** Run a human study correlating automated scores with human judgments of floor-plan correctness for a subset of apartments.
- **Fix the appendix:** Include the promised per-apartment results.
- **Add statistical tests:** Report p-values or confidence intervals for pairwise comparisons between models and baselines.
- **Define a proper random baseline:** Sample connectivity graphs from a null distribution matched to the ground-truth graph statistics (e.g., random spanning trees with correct room counts).
- **Evaluate humans on the full dataset:** Use the same 50 apartments, ideally with multiple human annotators to estimate human variance.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 2.0]
Average score: 1.3
Binary outcome: Reject
