## Summary

This paper introduces CERTAINLYUNCERTAIN, a 178K-sample benchmark for multimodal uncertainty awareness with a five-category taxonomy (epistemic: Knowledge, Complexity, Extraneous; aleatoric: Temporal, Ambiguity), a novel confidence-weighted accuracy metric, and extensive fine-tuning experiments across multiple VLMs and training strategies.

## Strengths

- **Comprehensive taxonomy and dataset scale**: The paper covers five fine-grained uncertainty types, substantially more than prior benchmarks like MM-Hal or UNK-VQA which cover 1-2 types (Table 2). The 178K sample size with contrastive pairs (answerable/unanswerable) is significantly larger than prior refusal-based benchmarks.

- **Empirical validation of the proposed metric**: Figure 4 demonstrates that confidence-weighted accuracy has stronger negative correlation with Expected Calibration Error (ECE) than standard LAVE accuracy, addressing a stated limitation of existing metrics that ignore confidence levels.

- **Broad experimental evaluation**: The paper evaluates across 7 benchmarks (refusal, hallucination, standard VQA), multiple model families (LLaVA variants, Qwen, InternVL, GPT-4V, Claude), and three training strategies (SFT, R-tuning, DPO). Table 6 shows improvements on UNK-VQA (41.32→59.70) and MM-Hal hallucination reduction (0.41→0.38) when the dataset is mixed with standard instruction data.

- **Contrastive data construction**: The inpainting-based pipeline (Figure 2) creates semantically aligned pairs where the same question is answerable on the original image but unanswerable on the perturbed version, forcing models to learn visual cues rather than dataset priors.

## Weaknesses

### Major

- **Overclaimed abstract statement about performance maintenance**: The abstract states fine-tuning "maintain[s] performance on standard VQA benchmarks," but Table 6 shows severe degradation when using the dataset alone: LLaVA-1.5-7B drops from 76.94 to 49.95 on VQAv2 (~35% relative drop), and Qwen-VL-Chat drops from 72.96 to 69.77. The claim only holds when the dataset is *mixed* with standard instruction data ("Ours+LLaVA"). This distinction is clarified in Section 3.3 (lines 302-303) but the abstract presents the dataset as a "plug-in improvement" without this critical caveat, which is misleading for readers who may only skim the abstract.

- **Confidence metric uses self-verification rather than generative probabilities**: The confidence-weighted accuracy metric (Equation 2, line 184) computes P(pred) by "prompting the model to verify if its own predicted answer is correct and extracting the probability of the 'yes' token" (line 186). This measures calibration of a separate self-verification step, not the generative output distribution. A model could be poorly calibrated generatively (high confidence in hallucinations) but well-calibrated in verification, yielding a high score on this metric while failing actual uncertainty awareness. This is a legitimate design choice the paper is transparent about (citing Whitehead et al., 2022), but it limits the metric's applicability and should be more explicitly framed as measuring "verification calibration" rather than generative calibration. Additionally, this approach cannot be applied to closed-source models (Table 4 shows "-" for GPT-4V and Claude-3.5 on this metric), limiting the benchmark's utility for the broader community.

### Minor

- **Inpainting artifact confound not fully addressed**: The "Extraneous" category (38% of the dataset) relies on inpainting to remove objects. The paper mentions a control experiment where random objects were inpainted and "performance did not fluctuate significantly" (line 133), suggesting models may be detecting visual artifacts rather than reasoning about missing semantic information. However, this control is not detailed in the paper (no quantitative results, no examples), and the concern that models are "gaming the benchmark" by detecting perturbation artifacts rather than epistemic uncertainty deserves more thorough investigation and discussion.

- **Taxonomy presentation creates confusion**: Figure 1 shows examples where some categories have definitive answers (e.g., Complexity Awareness: "What objects create a juxtaposition?" → "Airplane and barbed wires") while the text states these are categories where "admitting uncertainty... is the appropriate response" (line 85). The caption clarifies answers are "normalized to 'I don't know'" for simplicity (line 73), but the figure itself mixes IDK and non-IDK answers without clear labeling. This creates conceptual confusion about whether the taxonomy categorizes uncertainty *types* (with both answerable and unanswerable instances) or only unanswerable cases. The dataset does use contrastive pairs, but the figure presentation obscures this.

### Trivial

- **Statistical significance not reported for hallucination reduction**: Table 6 shows MM-Hal hallucination rate dropping from 0.41 to 0.38, but no variance or significance testing is provided. Given the small magnitude, this could be noise.

## Nice-to-Haves

- Add a control experiment for inpainting artifacts where questions *unrelated* to the inpainted region are evaluated on perturbed images to confirm models are detecting missing information rather than visual artifacts.

- Provide human validation statistics for the non-Extraneous categories (currently only the Extraneous test set underwent human filtering).

- Clarify in the abstract that performance maintenance requires mixing with standard instruction data, not using the dataset in isolation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Removed (Hard Rule - existence challenge)**: Harsh critic's implication that the metric "cannot be independently verified" for closed-source models. The paper explicitly acknowledges this limitation (Table 4 shows "-" for GPT-4V/Claude), and the paper's cited methods exist. This is a scope limitation, not an existence issue.

- **Removed (Misread - taxonomy)**: Critic's claim that Complexity Awareness example contradicts the definition. The figure caption (line 73) states answers are "normalized" for simplicity, and the dataset uses contrastive pairs (both IDK and non-IDK for each uncertainty type). The taxonomy categorizes uncertainty *types*, not only IDK cases. This is a presentation clarity issue, not a conceptual contradiction.

- **Removed (Scope creep)**: Request for confidence intervals on large-scale benchmarks where single-run evaluation is common practice in the VLM fine-tuning literature.

- **Removed (Strength Finder - generic)**: "This paper addressed an important problem" and "targeted an interesting question" - these are generic statements without concrete evidence.

- **Removed (Strength Finder - conflicts with verified weakness)**: Claim that the metric "effectively bridges the gap between predictive performance and model calibration" conflicts with the verified weakness that it measures self-verification rather than generative calibration.

- **Removed (Harsh critic - scope)**: Criticism that the paper "fails to discuss the distinction between selective prediction and uncertainty quantification." The Related Work section (lines 308-309) does discuss abstention and selective prediction approaches, distinguishing them from the paper's focus on epistemic/aleatoric uncertainty.

## Novel Insights

The paper's core insight—that VLMs need explicit training on uncertainty types rather than just general instruction tuning—is supported by the empirical results showing that "Ours-only" training dramatically improves uncertainty metrics while degrading standard VQA performance, but mixing recovers standard performance while retaining uncertainty gains. This suggests uncertainty awareness and standard capability are learnable separately and require balanced training. The contrastive pair design (answerable/unanswerable for similar visual contexts) is a genuinely useful contribution for teaching models to attend to specific visual cues rather than dataset priors.

## Suggestions

1. **Revise the abstract** to clarify that maintaining standard VQA performance requires mixing CERTAINLYUNCERTAIN with standard instruction data, not using it in isolation. The current phrasing is misleading.

2. **Reframe the confidence metric** more explicitly as measuring "self-verification calibration" rather than generative calibration, and discuss the implications of this design choice more thoroughly. Consider adding a comparison with generative token probability-based confidence where feasible (open-weight models).

3. **Add detail to the inpainting control experiment** mentioned in Section 2.2, including quantitative results and example images showing that models are not simply detecting artifacts.

4. **Clarify Figure 1** to distinguish which examples are IDK vs. non-IDK responses, or restructure to show contrastive pairs explicitly.

5. **Report statistical significance** for the hallucination reduction claims in Table 6.

## Calibration and Scoring

I retrieved calibration anchors across score ranges:

**High-scoring anchors (6.0-6.5):**
- *Post-hoc Probabilistic VLMs* (6.0): Training-free Bayesian uncertainty estimation with strong theoretical grounding and comprehensive experiments across 6 datasets. Stronger theoretical foundation than this paper.
- *Trading Visual Uncertainties* (6.5): Market-based multi-agent framework with theoretical proofs and 5 benchmarks. More novel framing.
- *Teaching VLMs to Admit Uncertainty* (6.0): OCR uncertainty tagging with GRPO and novel benchmark, but limited to synthetic data. More focused contribution.

**Medium-scoring anchors (4.5-5.5):**
- *AQuA* (5.5): 4-level ambiguity taxonomy with strategic responses. Similar taxonomy-based approach but smaller scale.
- *Human Uncertainty-Aware VQA* (5.0): Data selection framework with theoretical guarantees, limited to 2 datasets.
- *Confidence Calibration in VLA* (5.0): First systematic study but only OpenVLA evaluated, simulation-only.
- *Multi-modal Data Spectrum* (5.0): Large-scale empirical study across 23 benchmarks, analytical rather than innovative.

**Low-scoring anchors (≤4.0):**
- *Benchmarking Uncertainty Estimation in Science QA* (3.0): Large-scale but descriptive findings with critical issues ignored in main text.
- *Video Anomaly Detection Benchmark Analysis* (3.5): Framework analysis where conclusions were "widely acknowledged" without novel insights.

**Comparison:** This paper has stronger empirical scope than the 5.0 anchors (178K samples vs. smaller datasets, 7 benchmarks vs. 2-5, multiple model families vs. single model). However, it has more significant methodological concerns than the 6.0 anchors (abstract overclaim, self-verification metric limitations, inpainting confound). The paper is most comparable to *AQuA* (5.5), which also has a taxonomy-based approach with dataset construction concerns. The dataset scale and comprehensive evaluation push it slightly above the 5.0 anchors, but the methodological weaknesses prevent it from reaching the 6.0 tier.

**Final Score:** 5.5 (borderline Accept/Reject, similar to AQuA which was Accept Poster)

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>