Now I have enough information to write my final review. Let me synthesize all the information.

## Summary

TbLTA proposes the first weakly-supervised framework for dense long-term action anticipation, training exclusively from video transcripts (ordered action lists without temporal boundaries). It combines ATBA-based pseudo-label generation, CTC alignment, cross-modal attention, and a CRF decoder to produce frame-level future action predictions without dense frame-level annotations.

## Strengths

1. **Novel and significant problem formulation**: The paper identifies a clear gap—dense LTA has only been tackled with full supervision—and proposes the first fully transcript-only framework, establishing a new research direction. This is a genuine contribution regardless of architectural details.

2. **Well-motivated modular design**: The combination of ATBA alignment, CTC, cross-modal attention with local masking, CRF sequence modeling, and self-supervised duration priors addresses different aspects of the weakly-supervised LTA problem in a principled way. Ablations (Tables 3–4) show each component contributes meaningfully, with cross-modal attention providing up to ~5.7 points on Breakfast and CRF being critical for long-horizon predictions.

3. **Strong results on Breakfast**: Under 30% observation, deterministic TbLTA achieves 40.28% MoC at 10% prediction (outperforming all supervised baselines including ActFusion at 35.79%) and 29.03% overall average. This demonstrates that transcript supervision effectively captures procedural regularities in structured activities.

4. **Establishes first transcript-only LTA baseline**: Provides results on three standard benchmarks (Breakfast, 50Salads, EGTEA) with both deterministic and stochastic protocols, creating a reference point for future work.

## Weaknesses

### Major:

1. **Ambiguity in how Y_LTA and duration targets are derived under transcript-only supervision**: The CRF loss (Eq. 5–6) is defined over Z ∈ ℝ^{T_pred × |C|} with a target sequence Y_LTA of length T_pred, and the duration loss (Eq. 7) assumes per-segment indices i=1…T_pred with class labels y_i. Under pure transcript supervision, one only has a symbolic future sub-transcript (order of future actions), not a frame-resolved sequence of length T_pred. The paper does not explain how Y_LTA is constructed from the transcript: whether it is obtained by expanding the transcript sub-sequence into T_pred frames (e.g., proportional to predicted durations), or whether pseudo-labels are treated as ground truth. This ambiguity affects reproducibility and makes it unclear whether the CRF/duration losses are self-referentially training on pseudo-labels or genuinely using transcript-level information. While a careful reading suggests the ATBA module's pseudo-labels for the future segment (Ŷ_pred) serve as Y_LTA, this should be stated explicitly and the implications discussed.

2. **Deterministic results are not competitive on 50Salads**: At 20.92% average MoC (deterministic), TbLTA lags ActFusion by ~7.5 points (28.39%). The 50Salads dataset represents a harder setting with longer videos and denser action distributions. The paper frames this as a "complementary picture" but the gap is substantial for the claim that transcript-based supervision is a "robust alternative" to full supervision. The conclusion's assertion that "dense LTA does not need to rely on exhaustive frame-level annotation" is overstated given these results.

3. **Misleading claims of superiority that mix deterministic and stochastic protocols**: The abstract and conclusion state that TbLTA is "competitive with, and occasionally superior to, fully supervised approaches." The "superior" results on Breakfast (37.15% for TbLTA*-Top1 stochastic) are compared against deterministic supervised baselines (ActFusion 28.45%). The deterministic TbLTA (29.03%) is the fair comparison and is only modestly above ActFusion (28.45%). The paper acknowledges reporting both protocols but the main claims blend them rhetorically, overstating the comparison.

4. **No analysis of pseudo-label quality or error propagation**: The entire pipeline depends on ATBA-generated pseudo-labels. Without measuring their alignment accuracy (e.g., against ground truth on a validation set), it is impossible to distinguish whether method gains come from the architecture or from the quality of ATBA pseudo-labels. No ablation replacing ATBA with a simpler alignment (e.g., uniform segmentation) or using ATBA pseudo-labels directly with a simpler LTA decoder isolates the architectural contribution from the pseudo-labeling contribution.

### Minor:

1. **Section numbering and labeling is swapped**: Section 3.2.3 is labeled "Anticipation-Oriented Losses" but describes the ATBA alignment losses (L_A), while Section 3.2.1 is labeled "Alignment-Oriented Losses" but describes CRF and duration (L_LTA). The numbering also goes in reverse order (3.2.3 → 3.2.2 → 3.2.1). This is confusing though not fatal.

2. **Source of L_vid labels is unspecified**: The progressive training uses L_vid for pretraining (10 epochs), but it is not stated whether these come from the transcript-derived class set or from externally provided video-level labels. If external, this would be additional supervision beyond the transcript-only claim.

3. **Incomplete description of the cross-modal mask M**: The construction of the binary local mask M used in cross-attention is deferred to supplementary material, which limits reproducibility for a key architectural component.

4. **Duration modeling is weak**: The ablation shows the duration loss provides negligible benefit on 50Salads (~0.2 points) and only ~3.3 on Breakfast. The self-supervised class-averaged prior cannot capture instance-specific variations, and the paper acknowledges this as a remaining challenge.

5. **Limited EGTEA evaluation**: Only verb prediction is reported (not verb-noun), narrowing the scope of evaluation on this benchmark.

## Nice-to-Haves

- Pseudo-label quality analysis: Report ATBA alignment accuracy against ground truth on a validation split and analyze how errors propagate to anticipation.
- A simple baseline training a supervised LTA model (e.g., FUTR) on ATBA pseudo-labels as if they were ground truth, to isolate architecture vs. pseudo-labeling contributions.
- Analysis of robustness to transcript noise (missing actions, swapped order), since the practical argument for transcript-based supervision hinges on their scalability.
- Per-class/per-activity breakdown on 50Salads showing where transcript supervision fails and why.

## Removed Points

- **Harsh critic's claim of fundamental incoherence in CRF/duration losses, making the "transcript-only" claim invalid**: This overstates the issue. The CRF target Y_LTA is almost certainly derived from the ATBA pseudo-labels (which are generated from the transcript). While the paper should state this explicitly, the framework uses pseudo-labels as a bridge from transcripts to frame-level supervisions—this is standard practice in weakly-supervised learning, not a logical impossibility. The concern is about clarity, not fundamental flaw.

- **Demands for statistical variance (std over splits)**: Single-run evaluation is standard in this research community; requesting variance is nice-to-have but not a critical weakness.

- **Demand for computational cost analysis**: While useful, this is not standard in LTA papers and the framework inherits costs primarily from standard components (transformer encoder, ATBA).

- **Demand for a noise/transcript robustness experiment**: This is outside the paper's stated scope of demonstrating feasibility under clean transcript supervision. It would strengthen the practical argument but is not a core flaw.

- **Harsh critic's claim that the CTC application is unclear because it mixes observed and future frames**: The paper clearly states CTC is applied on segmentation head outputs over the full video during training. The CTC marginalizes over all alignments, which naturally accommodates variable durations—this is the standard CTC mechanism. The "hand-wavy" explanation could be improved but is not fundamentally unsound.

- **Demand for comparison with weakly-supervised TAS methods adapted for LTA**: The paper already compares with WS-DA (the only existing semi/weakly-supervised LTA method). Requiring TAS-to-LTA adaptations goes beyond standard practice and the paper's scope.

- **Claim that the training uses full video while supervised baselines may not, making comparison unfair**: The paper explicitly states this follows Gong et al. (2024) for ActFusion, which also trains on the full video. Supervised baselines in the comparison table use the same training protocol.

- **EGTEA restricted to verb prediction is a critical limitation**: This is appropriate given that the transcript contains action labels (verbs), making verb-only evaluation consistent with the supervision model. Not a flaw.

## Novel Insights

The key insight emerging from the reviews is the tension between weakly-supervised supervision pipelines that rely on pseudo-labels: the moment pseudo-labels serve as frame-level targets for downstream losses (CRF, duration), the method becomes a form of self-training rather than pure transcript-only supervision. This is not necessarily wrong—it is the standard pipeline in weakly-supervised learning—but the paper would benefit from acknowledging this more transparently. The method's effectiveness may be largely bounded by the quality of ATBA pseudo-labels, making pseudo-label analysis the most important missing experiment.

## Suggestions

1. Explicitly clarify how Y_LTA is constructed from the transcript/ATBA pseudo-labels in the CRF and duration losses—add a sentence or two specifying the mapping.
2. Report deterministic results prominently and separate deterministic vs. stochastic comparisons clearly; avoid claims of "superiority" when comparing across protocols.
3. Add a pseudo-label quality analysis against ground truth to establish the foundation of the pipeline.
4. Construct a simple "ATBA pseudo-labels → supervised LTA model" baseline to isolate architectural contributions.

## Score and Decision Calibration

Comparing against relevant papers:
- **AntGPT** (LTA, Accept poster, scores 5/8/6/6 ≈ 6.25): Novel LTA direction with LLMs, strong results but incremental novelty concerns.
- **GQgPj1H4pO** (Weakly-supervised VidSGG, Accept poster, scores 6/6/6/6 = 6): Novel weakly-supervised direction, pipeline using off-the-shelf components, similar incremental novelty concerns.
- **DE2RMJVjgI** (Weakly-supervised with pseudo-labels, Reject, scores 5/3/6/3 ≈ 4.25): Incremental novelty on existing pipeline, weak experiments.
- **Actions-to-Action** (LTA, Reject, scores 5/3/3/6/5 ≈ 4.4): Weak novelty, overselling claims, limited insight.

This paper is similar to the GQgPj1H4pO paper in establishing a new weakly-supervised direction with a pipeline of existing components, but with more original integration and stronger results on at least one benchmark. However, it has more serious clarity issues (Y_LTA definition, swapped sections) and overclaims about competitiveness. It falls between the AntGPT/GQgPj1H4pO level (6ish) and the DE2RMJVjgI level (3-5). The problem formulation is genuinely novel and the results on Breakfast are meaningful, but the presentation gaps and overstated claims are real concerns.

Score: 5.5 — the paper addresses an important and underexplored problem with promising early results, but has significant clarity issues in the loss formulation, overclaims about competitiveness, and lacks crucial supporting experiments (pseudo-label quality, simple baselines).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>