Now I have enough information to write the final review. Let me synthesize everything.

## Summary

Blueprint-Bench introduces a benchmark that evaluates spatial reasoning in AI models through the task of converting apartment photographs into 2D floor plans. The benchmark is model-agnostic, scoring any system that generates an image from images via a computer-vision-based extraction pipeline that produces a connectivity graph and size rankings, then computes a composite similarity score against ground truth. Across 12 models spanning LLMs, image generation models, and agents, most perform at or below a no-vision-input baseline, while humans substantially outperform all models, revealing a spatial reasoning blind spot in current generalist AI systems.

## Strengths

- **Novel and well-motivated task design**: The photo-to-floor-plan task is an inventive probe of spatial intelligence. As the paper argues in Section 1, it uses in-distribution inputs (photographs) but requires genuine spatial reconstruction—a capability gap worth measuring. The task is intuitive, meaningful, and requires integrating partial visual information into a coherent spatial representation.

- **First cross-architecture comparison of spatial intelligence**: The paper evaluates 12 systems across three architectures (LLMs, image generation models, agents) on the same task with the same metric. This enables meaningful comparisons that, as the paper claims, have not been done before. The finding that agents with iterative refinement don't improve over single-pass generation (Section 3, Figure 8) is a concrete and informative result.

- **Transparent failure-mode analysis**: The paper distinguishes instruction-following failures from genuine spatial reasoning deficits. GPT-Image "consistently outputs floor plans mostly according to the rules" yet scores at the random baseline (Section 3), proving the benchmark can detect real spatial deficits. The trace analysis showing Codex never inspects its output while Claude Code iterates but still fails (Figure 8) adds mechanistic insight beyond aggregate scores.

- **Honest reporting of alternatives tried**: Section 2.4 documents rejected alternatives (LLM-based extraction, shape-matching) with concrete reasons, demonstrating thoroughness in benchmark design. The paper is transparent about limitations rather than hiding them.

- **Open-source evaluation code with private dataset**: Releasing scoring code while keeping most of the dataset private balances reproducibility with benchmark integrity (Reproducibility Statement).

## Weaknesses

### Fatal
None.

### Major

- **The evaluation metric measures connectivity and size ranking, not geometric fidelity, creating a gap between what is claimed and what is measured**: The paper claims to evaluate "spatial intelligence" defined as "inferring room layouts, understanding connectivity, and maintaining consistent scale" (Abstract), but the metric (Section 2.3) only measures connectivity graph properties (edge overlap 50%, degree correlation 20%, density 10%) and size ranking (room count 10%, door count 5%, door orientation 5%). No component measures room positions, shapes, wall placements, distances, or layout accuracy. A model producing a spatially nonsensical floor plan with correct connectivity could score highly; conversely, a model with approximately correct spatial layout but misranked room sizes gets double-penalized (size rank errors propagate into connectivity matching). The paper acknowledges this in Section 2.4 but frames it as a minor limitation rather than a fundamental gap—the metric covers only a subset of the spatial properties the paper claims to evaluate. The paper's own attempted shape-matching alternative was abandoned because it "harshly penalized small mistakes in unpredictable ways," which suggests the current metric was chosen for robustness rather than validity.

- **Instruction-following conflation undermines scores for a subset of models**: The paper acknowledges "Blueprint-Bench should test spatial intelligence, not instruction following" (Section 2.4), yet models failing the 9 formatting rules receive scores uninformative about their spatial reasoning. GPT-4o (0.15) and NanoBanana (0.18) are explicitly attributed to instruction-following failures (Section 3). Their actual spatial reasoning ability is unmeasured. While GPT-Image demonstrates that the benchmark can detect genuine spatial deficits when instructions are followed, the headline rankings in Figure 5 conflate two distinct failure modes, making the comparative results misleading for 2 of 12 models. The paper's defense ("at current model capabilities, we think this is the right tradeoff") is an unsupported assertion with no analysis of how much the rankings shift when instruction-following failures are excluded.

### Minor

- **The "random baseline" is misleadingly labeled**: Section 2.2 describes creating "a worst-case baseline by generating typical floor plans using LLMs and image generation models without any image input," but figures label this as "Random baseline." This is not a truly random baseline—it reflects model-generated floor plans with structural regularities typical of apartments. While generating without visual input is a reasonable baseline, calling it "random" is misleading. The paper's central finding ("most models perform at or below a random baseline") is more accurately stated as "most models perform at or below a no-vision-input baseline." The distinction matters because model-generated typical floor plans will share structural regularities with ground truth (e.g., typical apartment topologies), setting a higher bar than true randomness.

- **Composite score weights lack justification or sensitivity analysis**: The weights (50% edge overlap, 20% degree correlation, 10% density, 10% room count, 5% door count, 5% door orientation) are presented without justification (Section 2.3). Given that these weights determine model rankings, even a brief sensitivity analysis showing robustness to weight perturbations would strengthen the benchmark's credibility.

- **Human baseline limited to only 12 of 50 apartments**: Figure 7's human comparison uses only 12 apartments, making it difficult to generalize the human-AI gap. The paper is transparent about this but does not discuss why more apartments weren't evaluated or how representative these 12 are.

### Trivial
- The 2.5 standard deviation error bars in Figure 7 are non-standard (typically 1 or 2 SD), making the error bars large and potentially masking how close some model scores are to human performance on this subset.

## Nice-to-Haves

- Adding even a simple geometric metric (e.g., IoU of room regions after optimal alignment, or Hausdorff distance between room boundaries) alongside the connectivity metric would substantially strengthen the "spatial intelligence" claim and provide a more complete picture.
- Reporting individual sub-scores (edge overlap, degree correlation, etc.) separately would reveal where models specifically fail and help decompose the composite score into interpretable components.
- A true random baseline (random graphs matching dataset room/door distributions) alongside the no-vision-input baseline would clarify whether the current baseline is inflated by structural regularities.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Model categorization errors in Figure 5 (Claude Opus 4.1 and Sonnet 4 as "Image model")**: This appears to be a parser artifact. The original figure likely uses visual patterns (striped vs. dotted bars) to distinguish model types, but the alt-text table generated by the parser incorrectly categorized LLMs as "Image models." The paper clearly discusses three categories (LLMs, image generation models, agents) in the text, so this is not an author error.

- **Different model names in appendix vs. main text (reproducibility concern)**: The appendix references "Claude 4.5" and "Claude 3.5 Sonnet" while the main text uses "Claude Opus 4.1" and "Claude Sonnet 4." This is an internal consistency issue, but falls under reproducibility/implementation details that should not be flagged per the rules. It could also result from iterative model updates during the review process.

- **Figure 5 vs. Figure 7 showing slightly different scores for same models**: The scores differ because Figure 5 covers all 50 apartments and Figure 7 covers only 12 (with human baselines). This is explained in the paper and is not an inconsistency.

- **Formatting rules (3-pixel-wide lines, 10x10px dots, pure colors) creating unfair burden for image models**: The paper acknowledges the instruction-following burden in Section 2.4 and the formatting rules serve a legitimate purpose (enabling robust CV-based scoring). This is a design tradeoff, not a flaw.

- **Missing ablation on input type (text descriptions, subset of images, room count hints)**: This would be informative but goes beyond the paper's stated scope of comparing spatial intelligence across model architectures.

- **Missing human evaluation of spatial plausibility**: Valuable but not standard for this type of benchmark paper.

## Novel Insights

The paper's most insightful observation is the dissociation between iterative refinement capability and spatial reasoning improvement: Claude Code iterates and refines its output (Figure 8) yet still performs near the no-vision-input baseline, while Codex generates in one pass and achieves comparable or better scores. This suggests that current agents lack an internal model of spatial structure that would allow meaningful self-correction—iterating without a spatial "ground truth" internal representation is no better than generating once. This has implications beyond this benchmark: it indicates that agentic refinement loops may not help for tasks where the agent lacks foundational understanding of the domain.

## Suggestions

- Rename the "Random baseline" to "No-vision-input baseline" or "Prior-only baseline" throughout the paper and figures. This is a simple change that would make the central claim more accurate and defensible.
- Add a sensitivity analysis on the composite score weights (±10% perturbation) and report how rankings change. This would take minimal effort and significantly strengthen confidence in the benchmark's rankings.
- Consider reporting sub-scores separately for each model in a table, so readers can see whether models fail at connectivity inference, size ranking, or other specific components.

## Score and Decision

**Calibration anchors used:**

- **SPACE (WK6K1FMEQ1.md, avg 6.75, Accept Poster)**: Tests spatial cognition in frontier models across 15 tasks grounded in cognitive science, with human baselines. Blueprint-Bench is weaker because its metric doesn't fully measure what it claims, has fewer tasks, and a limited human baseline, but is stronger in cross-architecture coverage (including image models and agents).

- **Symbolic Graphics Programs (Yk87CwhBDx.md, avg 7.33, Accept Spotlight)**: Novel benchmark with procedural generation, clear evaluation metrics, and a proposed improvement method (SIT). Much more complete evaluation pipeline than Blueprint-Bench.

- **VLM 3D Spatial Reasoning (uBhqll8pw1.md, avg 4.00, Reject)**: Tests VLMs on spatial reasoning with similar concerns about whether the benchmark measures what it claims (tasks may be solvable by text alone). Blueprint-Bench has more model coverage and a more novel task but similar metric validity concerns.

- **KiVA (vNATZfmY6R.md, avg 7.00, Accept Poster)**: Visual analogy benchmark with 4,300 tasks, 3-stage evaluation, grounded in developmental psychology. More comprehensive than Blueprint-Bench but less cross-architecture diversity.

- **MPG (MGceYYNvXp.md, avg 1.50, Reject)**: Proposes metrics that don't validly measure claimed capability. Blueprint-Bench is clearly better than this—its metric captures real (if incomplete) spatial properties.

- **Graph Reasoning (S9YfP4rsfX.md, avg 2.50, Reject)**: Limited scope benchmark with somewhat expected results. Blueprint-Bench has more novelty but partially similar concerns.

Blueprint-Bench sits between the low-scoring papers with fundamental metric validity problems (1.5-2.5) and the medium-high spatial reasoning benchmarks (4.0-7.0). Its metric captures real spatial properties (connectivity, size ranking) but misses geometric fidelity—a significant gap that undermines the "spatial intelligence" framing but doesn't invalidate the entire benchmark. The cross-architecture comparison and novel task design add genuine value. Compared to the VLM 3D spatial reasoning paper (4.00), Blueprint-Bench has comparable metric validity concerns but more novelty and model coverage. Compared to SPACE (6.75), it has a narrower scope and weaker metric alignment with its claims. A score of 4.5 reflects these tradeoffs: the paper makes a real contribution but the metric gap and baseline labeling issues are significant enough to keep it below the acceptance threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>