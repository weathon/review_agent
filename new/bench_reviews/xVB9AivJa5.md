Now let me run calibration searches to anchor my scoring.Now I have sufficient information to write the final consolidated review.

---

## Summary

Blueprint-Bench introduces a benchmark for spatial reasoning in which AI systems must convert a sequence of apartment photographs into a 2D floor plan conforming to nine strict formatting rules. The benchmark evaluates four LLMs, two image generation models, and two agent scaffolds on 50 apartments, scoring outputs with a graph-based similarity metric that measures room connectivity and size-ranking agreement against ground-truth floor plans. The central finding is that most models perform at or below a randomly generated baseline, while human performance remains substantially superior, suggesting a spatial-reasoning blind spot analogous to ARC.

---

## Strengths

- **Genuinely novel task framing**: Using in-distribution photographs as input but requiring out-of-distribution spatial reconstruction is a well-motivated probe of spatial intelligence, drawing a clear conceptual parallel to ARC (Section 1). This is more ecologically valid than grid-pattern tasks.
- **Cross-architecture comparison**: Evaluating LLMs, native image-generation models, and agent scaffolds on the same task and metric is a concrete contribution; the paper's claim that this enables the first direct numerical comparison between image generation models and their underlying LLMs (Section 1, lines 156–161) is plausible and useful.
- **Principled, automated scoring**: The graph-based metric (Jaccard edge overlap, degree correlation, density, room count, door count, orientation) operating on deterministic CV extraction is sounder than pixel-level similarity or LLM-based extraction, the latter of which the paper empirically shows to be unreliable (Section 2.4).
- **Honest limitation discussion**: The paper candidly identifies the instruction-following confound (Section 2.4: "Blueprint-Bench should test spatial intelligence, not instruction following") and reports empirical failures of alternative extraction strategies.
- **Open-source code and living leaderboard**: Commitment to community submissions and ongoing evaluation adds practical value.

---

## Weaknesses

### Fatal
None.

### Major

- **Instruction-following conflation undermines the headline claim for a subset of models.** The paper's central claim—"most models perform at or below a random baseline" as evidence of a "significant blind spot in spatial intelligence"—is not fully supported for NanoBanana and GPT-4o. The paper itself attributes their near-zero scores to instruction-following failure (Section 3: "this can be attributed to poor instruction following, leading to outputs that do not adhere to the rules"). For these models, we have no evidence about their spatial reasoning ability, only that they cannot follow formatting constraints. The paper acknowledges this in Section 2.4 but offers no methodological remedy, making the spatial-intelligence conclusion unreliable for at least two of the eight evaluated systems. A component-level breakdown (spatial metrics vs. formatting compliance) would make this distinguishable.

- **Random baseline is inadequately specified.** The paper states (Section 2.2) that a "worst-case baseline" was created "by generating typical floor plans using LLMs and image generation models without any image input," and Figure 5 plots this as the black horizontal reference line. However, no further detail is given: which models, how many samples, what prompts, what score distribution. Given that this baseline is the comparison point for the paper's core conclusions ("statistically perform better than the random baseline" for four models), the lack of transparency makes those conclusions difficult to interpret. A model scoring above a high-variance random baseline is very different from one scoring above a near-zero baseline.

- **Agent comparison confounds model and scaffold simultaneously.** The paper concludes that "iterative refinement capabilities show no meaningful improvement," but the two agents tested—Codex CLI with GPT-5 and Claude Code with Claude 4 Opus—use different base models in addition to different scaffolds. Furthermore, the paper itself notes that Codex CLI did not exploit its iterative capability (Section 3: "It never even looked at the image it created before submitting"), effectively running as a single-pass LLM. The valid conclusion is that, under these specific configurations, iterative refinement did not help—not that iterative refinement in general is ineffective. A same-model, with/without-iteration comparison is missing.

### Minor

- **Thin human baseline.** The human performance data comes from 12 of 50 apartments with an apparently single human rater (Section 3, Figure 7 caption). The conclusions drawn—"humans remain substantially superior," "all human floor plans were drawn such that the connectivity between the rooms was correct"—rest on 12 data points. With multiple raters and the full 50-apartment set, these claims could be made robustly; as stated, they provide only weak calibration.

- **Unvalidated scoring weights.** The composite score uses weights (50% edge overlap, 20% degree correlation, 10% density, 10% room count, 5% door count, 5% door orientation) without justification or ablation. The paper does not show that model rankings are stable under alternative weighting schemes. This matters because the "statistically better than random baseline" determination for borderline models (e.g., GPT-5-mini) could be sensitive to weight choices.

- **Statistical claims without methodology.** The paper states several models "statistically perform better than the random baseline" (Section 3) but does not describe the statistical test, significance level, or whether corrections for multiple comparisons were applied.

- **SVG vs. raster output confound.** LLMs generate SVG code that is rendered; image models generate raster images directly. SVG generation requires knowledge of coordinate systems and path syntax beyond spatial reasoning. The paper does not demonstrate that SVG coding ability is not a bottleneck for LLM scores.

### Trivial

- The dataset adaptation process (converting commercial floor plans to the 9-rule format) is described only briefly. The number of apartment listings processed, and what human effort was involved, is unclear.

---

## Nice-to-Haves

- A per-component score breakdown per model (edge overlap vs. room count vs. door accuracy separately) would help distinguish spatial failures from formatting failures and directly address the instruction-following confound.
- An ablation holding the model constant and varying the scaffold (single-pass API call vs. agent loop) would provide cleaner evidence on iterative refinement.
- Including 3–5 human raters on all 50 apartments and reporting inter-rater agreement would validate both the scoring algorithm and the human-AI gap claim.
- Examples showing high-scoring outputs (e.g., score ~0.6) alongside low-scoring ones would help readers calibrate what the metric measures qualitatively.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"First numerical framework" claim is overclaiming** (harsh critic, §Abstract): The critic suggests a rich literature of spatial benchmarks exists, but per rules I cannot verify missing related works. Not removed for this reason, but the specific claim is scoped to cross-architecture LLM–vs.–image-model comparison and is defensible as written.

- **Ground truth adaptation process is non-reproducible** (harsh critic, §2.1): The critic calls this a "reproducibility concern." Keeping 38/50 apartments private to prevent overfitting is a legitimate design choice; this is not a methodological flaw. Removed as a nitpick on reproducibility.

- **"NanoBanana's spatial layout might be correct; we have no way to know"** (harsh critic): Partially absorbed into the major weakness above. The extreme version of this critique—that the paper's *entire* headline conclusion is invalid—is overstated; the LLM results (which largely follow formatting rules) do support the conclusion. The weaker form is retained.

- **Claude's "Each room is fully enclosed" anecdote is a single data point** (harsh critic): Valid but minor; the paper presents it as illustrative, not as a systematic finding. Removed as a severity-inflated nitpick.

- **Novelty claim too strong / missing related work** (harsh critic): Cannot verify existence of specific missing references; removed per hard rules.

---

## Novel Insights

Blueprint-Bench surfaces an underappreciated methodological tension in spatial reasoning benchmarks: strict output formatting, which enables automated, robust scoring, simultaneously makes it impossible to attribute failure to the cognitive capability being tested (spatial reasoning) versus the format specification (instruction following). This is not unique to Blueprint-Bench—it likely affects many structured-output benchmarks—but the paper's explicit acknowledgment and failure to resolve it points to a broader need for two-stage evaluation designs: first check format compliance, then score spatial structure only on compliant outputs. Additionally, the observation that the iterative agent (Claude Code) produces substantially worse *first* drafts than the non-iterating single-pass system (Codex CLI), yet ultimately achieves similar final scores, hints that iterative refinement may be compensating for a weaker initial prior rather than genuinely improving spatial reasoning—a distinction worth investigating.

---

## Suggestions

1. Report a per-component score breakdown per model to separate spatial from formatting failures.
2. Clearly define the random baseline: which models, how many outputs, what prompts, and what score distribution.
3. Conduct the within-model iterative-vs.-single-pass ablation using the same underlying model (e.g., Claude 4 Opus via API in both modes).
4. Expand the human evaluation to at least 3 raters on all 50 apartments and report inter-rater agreement.
5. Provide a justification or sensitivity analysis for the composite score weights.
6. Report the statistical test methodology (test type, significance threshold, multiple-comparison correction) for the "statistically better than random" claims.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|---|---|---|
| SPACE (WK6K1FMEQ1) | 6.75 (Accept) | Most similar topic (spatial cognition in frontier models); more comprehensive (15 tasks, cognitive science grounding, full human baselines), better methodology. Blueprint-Bench is narrower and has more methodological gaps. |
| 3D-PC (UIFAJZ22ZF) | 6.67 (Accept) | Visual spatial reasoning benchmark; more focused but rigorous. Blueprint-Bench lacks equivalent methodological rigor. |
| PolyMATH (WVBzN1HIFS) | 5.50 (Reject) | Multi-modal reasoning benchmark; 5,000 images, 15 models, 4 prompting strategies — more comprehensive despite being rejected. Blueprint-Bench is less comprehensive but has a more novel task framing. |
| VLM 3D Indoor (uBhqll8pw1) | 4.00 (Reject) | Evaluates VLMs on spatial layout reasoning; rejected for motivation issues, thin dataset, and overclaiming. Comparable scope and methodological weaknesses to Blueprint-Bench. |
| Floor plan diffusion (skJLOae8ew) | 3.00 (Reject) | Floor-plan generation with diffusion; rejected for minimal technical novelty and insufficient evaluation. Blueprint-Bench has substantially more intellectual content. |

**Positioning:** Blueprint-Bench sits between the VLM-3D-Indoor paper (4.00) and PolyMATH (5.50). It has a more novel and well-motivated task framing than the 3.00 floor-plan paper, and its explicit acknowledgment of limitations is a positive. However, the three major weaknesses—the instruction-following confound that partially undermines the central claim, the ill-defined random baseline against which all results are calibrated, and the thin human evaluation—are genuine methodological gaps. The paper is closer in rigor to the 4.00 anchor than the 5.50 anchor. The interesting task framing and honest limitation discussion push it slightly above 4.0, but the core claim is not fully supported by the data.

**Final score: 4.0**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>