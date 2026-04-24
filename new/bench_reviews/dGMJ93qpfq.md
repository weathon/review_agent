## Summary

This paper proposes Patch-Aware Prompting (PAP), a modular framework for vision-language prompt tuning that injects patch-level local semantics into three branches: vision features (via intra/inter-view patch consistency losses), text prompts (via clustering-based view-tailored biases), and predictions (via KL-divergence across views). The method is designed as a plug-in module for existing prompt-tuning pipelines and is evaluated on base-to-novel generalization, cross-dataset transfer, and domain generalization benchmarks.

## Strengths

- **Clear, relevant motivation and modular design.** The paper targets a genuine limitation in prompt tuning—over-reliance on global features—and proposes a concrete multi-branch mechanism to leverage patch-level information. The plug-in design is validated by applying PAP to PromptSRC, DePT, CoCoOp, and CoPrompt (Tables 1, 11).
- **Extensive multi-benchmark evaluation.** The paper evaluates across 11 datasets for base-to-novel transfer, 10 target datasets for cross-dataset evaluation, and 4 ImageNet variants for domain generalization, providing broad empirical coverage.
- **Ablation studies supporting design choices.** Tables 4–12 provide granular ablations over losses, clustering methods, projection blocks, crop strategies, and augmentations, showing that the full configuration performs best among the tested variants.

## Weaknesses

### Fatal
None

### Major

- **Confounded capacity baseline undermines the central claim.** PAP increases learnable parameters by roughly an order of magnitude (PromptSRC: 0.46M → 4.89M; DePT: 0.74M → 5.12M) and introduces new architectural components (text adapter, convolutional projection block). Table 9 shows that adding a text adapter alone provides much of the base-class gain (84.40 vs. 83.24 base accuracy), yet the paper never isolates a matched-capacity baseline that includes these extra parameters and compute *without* the patch-level consistency losses or Voronoi clustering. Because the final improvements are small (often ≈1% HM or less), the current experiments cannot disentangle whether the gains stem from patch-level semantics or simply from increased model capacity and additional views.
- **Sub-1% margins reported without variance.** The paper reports single-run results with no standard deviations or multi-seed statistics. On base-to-novel, cross-dataset, and domain-generalization tasks, many improvements over strong baselines are <1%, making them difficult to assess for statistical significance.

### Minor

- **Ambiguous hyperparameter selection protocol.** The paper states that loss weights λ_p, λ_t, λ_l are set to defaults but “modify it for individual dataset when required” (Section 4). It does not describe how these per-dataset modifications are chosen—e.g., via a held-out validation split, base-class performance, or another protocol—which limits the credibility of the small reported margins.
- **Vague specification of the Voronoi clustering step.** The text-prompt generation mechanism is described only as “clustering the vision zero-shot patch features P̄ using the Voronoi algorithm” (Eq. 9). The paper does not specify how generator points are chosen, whether clustering operates on spatial coordinates or feature embeddings, or how the bias vectors are derived from the cells. This omission makes the clustering step difficult to reproduce and the comparison with KMeans/EM in Table 8 hard to interpret.
- **Overstated novelty in the abstract.** The abstract claims “the first integration of such semantics in this context,” whereas Section 2 acknowledges Long et al. (2024) as an independent concurrent work that also uses clustered patch tokens for text prompts. The abstract should be revised for accuracy.

### Trivial

- Table 2 reports two separate values in the “Ave.” column without defining what each represents.
- Table 1 marks CoPrompt and DePT with * to indicate reproduction, but does not do so for PromptSRC, leaving it unclear whether those numbers are reproduced or taken from the original paper.

## Nice-to-Have

- A direct experimental comparison to Long et al. (2024) would strengthen the novelty claim.
- Qualitative visualizations of the patch clusters (spatial overlays or nearest-neighbor retrievals) would validate that text prompts are conditioned on meaningful local information.
- Reporting mean and standard deviation across at least three random seeds for ImageNet and one fine-grained dataset.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Criticism about Eq. 12 introducing an undefined symbol P̄.** This is factually wrong: P̄ is defined in Eq. 9 as the vision zero-shot patch features used for clustering.
- **Domain-generalization criticism regarding DePT.** The harsh reviewer claims that source-domain accuracy rises over DePT by 0.72% in the domain-generalization benchmark. DePT is *not* included as a baseline in Table 3, so this comparison is fabricated.
- **“Table 10 reproduces a standard multi-crop result.”** This is a strawman; the paper presents the crop ablation as a design-choice analysis, not as a novel scientific finding.
- **Parser-related notation and formatting nitpicks.** Several equation and symbol complaints appear to be exacerbated by PDF-to-text artifacts (e.g., Eq. 5’s duplicated term). Per the review rules, these are not author errors.

## Novel Insights

None beyond the paper’s own contributions.

## Suggestions

1. **Add a matched-capacity baseline.** Run PromptSRC/DePT equipped with the text adapter, convolution projection, and multi-view augmentation but *without* patch consistency losses or Voronoi clustering. This is the single most important experiment needed to isolate the contribution of patch-level semantics.
2. **Clarify the Voronoi clustering implementation.** Describe generator initialization, the space in which clustering occurs (feature vs. spatial), and how bias vectors are extracted from cells.
3. **Report variance and the hyperparameter selection protocol.** State explicitly how per-dataset loss weights are chosen and provide standard deviations across seeds.

## Score and Decision

**Score: 5.0**

**Calibration comparison:**
- **CLIPSelf** (`/home/wg25r/review_agent/human_reviews/DjzvJCRsVf.md`, avg 7.0, Accept spotlight): Conceptually simple, large and clear gains, extensive dense-prediction evaluation, no confounds. The current paper is well below this standard due to its confounded capacity increase and much smaller margins.
- **CARPRT** (`/home/wg25r/review_agent/human_reviews/fRpAUgKJhT.md`, avg 5.75, Reject): Similar marginal performance gains (<1%) and computational concerns, but theoretically well-motivated. The current paper has broader evaluation and a more interesting architectural idea, yet suffers from a more severe experimental confound.
- **BlzBcWYmdB** (`/home/wg25r/review_agent/human_reviews/BlzBcWYmdB.md`, avg 5.0, Reject): Extensive theoretical analysis but thin empirical verification; rejected. The current paper is the mirror image—extensive experiments but a key missing control—placing it at the same 5.0 level.
- **DIP** (`/home/wg25r/review_agent/human_reviews/SBZiZFp560.md`, avg 4.33, Reject): Confounded optimization objective and missing baselines. The current paper is more coherent but has a similar fundamental issue (capacity confound), keeping it above the very low band but below the acceptance threshold.

The paper presents a plausible and modular idea with broad empirical coverage, but the lack of a matched-capacity baseline means its central claim—that patch-level semantics drive the improvements—is not credibly supported. Combined with small unvariance-reported margins and ambiguous methodological details, the submission is below the acceptance threshold.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>