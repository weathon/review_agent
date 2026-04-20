## Summary

This paper identifies spurious attributes ("black sheep") in vision-language model few-shot adaptation—attributes that co-occur with categories but aren't intrinsic parts of them—and proposes two solutions: SAP, which uses MLLMs and CBMs to filter spurious attributes; and SAS, which introduces pseudo-categories defined by spurious attributes to train a subsidiary distinguishing task. The approach shows consistent accuracy improvements across 11 datasets and 3 generalization tasks while maintaining base performance.

## Strengths

- **Compelling empirical validation across diverse settings**: Figure 3 demonstrates consistent out-of-distribution accuracy gains across 11 base-to-new datasets, cross-dataset transfer, and domain generalization tasks. The improvements hold across multiple PEFT families including prompt tuning (CoCoOp, PromptSRC), attribute-based methods (CPL, ArGue), and adapters (CLIP-Adapter), validating the plug-and-play claim.

- **Creative counter-group evaluation**: Table 2's adversarial evaluation with spurious-attribute-filtered test images shows SAS achieves disproportionately larger gains on the counter group (e.g., +6.48% for CPL on FGVCAircraft) compared to the standard test set (+0.21%), providing evidence that gains come from reducing spurious reliance rather than mere data augmentation.

- **Adaptive threshold strategy with ablation**: SAP's adaptive thresholding (γ_c set as minimum core attribute weight) outperforms all fixed thresholds in Table 4 (HM 80.38 vs 79.81 best fixed), and the ablation shows performance degrades when γ is too high or too low, confirming the importance of proper spurious attribute selection.

- **Practical efficiency consideration**: Table 5 demonstrates the selective optimization trick reduces training time by ~20% while preserving most accuracy gains, making the method practically scalable.

- **Well-motivated problem identification**: Table 1 convincingly shows that removing <7% of spurious attributes yields significant generalization improvements (CPL: 65.30% → 67.66%, ArGue: 66.07% → 67.69%), and the CBM analysis in Figure 1 reveals that spurious attributes dominate top-3 influential attributes.

## Weaknesses

### Fatal

None

### Major

- **Mechanism claim vs. actual method mismatch**: SAS is positioned as "shielding" or "mitigating" spurious feature reliance (Section 3.4), but the subsidiary task is simply a standard cross-entropy over an extended label space (Equation 3). Training the model to classify images with spurious attributes into separate pseudo-categories does not inherently penalize reliance on those features through gradient reversal, adversarial alignment, or GroupDRO-style reweighting. The model may still use spurious features for the subsidiary task or develop new shortcuts. The paper's claim that SAS "reduces reliance on spurious attributes" is an interpretation not guaranteed by the optimization objective. A more honest framing would be that SAS provides additional contrastive supervision; the spurious-mitigation mechanism is inferred from empirical results rather than structurally enforced.

### Minor

- **CBM few-shot reliability for attribute probing**: The SAP method trains a CBM on 16-shot data to derive the weight matrix $\mathcal{W}$ (Section 3.3). In such a data-limited regime, the linear head will naturally absorb the strongest statistical correlations present in the training sample. This means high-weight non-core attributes are the most predictive features—which may be valid contextual cues rather than spurious correlations. The paper does not address how CBM overfitting in few-shot settings might affect attribute identification reliability.

- **Evaluation lacks established spurious-correlation benchmarks**: While the 11-dataset evaluation is comprehensive for general PEFT benchmarking, spurious-correlation research has established standards (Waterbirds worst-group accuracy, CelebA subgroup performance) that directly measured attribute reliance. The counter-group evaluation, while creative, is an ad-hoc filter based on semantic similarity rather than a controlled distribution-shift benchmark. Comparisons with spurious-correlation-specific baselines (e.g., GroupDRO, CORAL) are noted as missing.

- **Selective optimization criterion underspecified**: Table 5 introduces a "selective trick" that optimizes only ~10% of categories, but the main text does not clearly specify how these categories are selected. The supplementary materials reference suggests it uses the SAP-derived weights, which would create circularity: categories selected by the spurious identification mechanism are the only ones used to validate the method.

### Trivial

- **Saliency map interpretability limitations**: Figure 5 relies on qualitative saliency maps to claim SAS "shifts attention from spurious attributes to target objects." While these provide visual intuition, saliency maps are known to be unstable and do not quantitatively measure feature attribution. Quantitative concept probing scores would strengthen the evidence.

## Nice-to-Haves

- Provide quantitative linear probing results on spurious vs. core attribute features before and after SAS training to objectively demonstrate reduced reliance on spurious pathways.
- Report compute cost and API call volume for MLLM-based SAP across the 11 datasets to help practitioners assess practical feasibility.
- Control experiments replacing pseudo-categories with random categories or core-attribute images to isolate the spurious-mitigation effect from general regularization benefits.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **Related work coverage**: The harsh critic claims the paper misses TCAV, automated concept discovery, causal concept discovery, invariant representation learning, causal fine-tuning, and VLM-specific debiasing. However, the paper does reference concept bottleneck models (Koh et al.), spurious attribute identification literature (Singla & Feizi, Adila et al.), and spurious correlation mitigation approaches (domain mix-up, instance reweighting, contrastive learning). The scope is reasonable for the claimed contribution; missing additional references is not a substantive weakness.

2. **"Structurally incapable" claim about SAS**: The harsh critic claims the cross-entropy auxiliary loss is "structurally incapable of mitigating spurious feature reliance." This overstates the case. While SAS doesn't use adversarial objectives, the subsidiary classification task does create additional decision boundaries that force the model to distinguish target categories from spurious-attribute-defined categories. The mechanism may be simpler than claimed but is not invalid.

3. **Reproducibility concerns about model release**: The paper states "code will be available" and uses established methods (GPT-4V, Stable Diffusion, CLIP). These are all accessible tools; no claims about unreleased or unverifiable systems are present.

4. **Hyperparameter details for CBM training**: The CBM uses established methodology (Koh et al. 2020, Yang et al. 2023) with dot-product scoring and linear projection. The few-shot concern is valid but the specific training details are implementation details appropriate for the supplementary material.

5. **Architectural implications of SAS**: The claim that SAS "inherently expands the classification label space" is true but the paper acknowledges this and frames it as introducing pseudo-categories alongside real ones. This is standard in multi-task learning; gradient interference is managed through the loss weighting scalar $\lambda$.

6. **Missing standard spurious-correlation baselines as unfair comparison**: The paper compares against standard PEFT methods within its scope. Not including GroupDRO or CORAL is a limitation but not an unfair comparison asymmetry favoring the proposed method.

## Novel Insights

This paper identifies a genuine and practical issue in VLM adaptation: spurious attributes that co-occur with categories but aren't intrinsic parts of them can disproportionately influence model decisions despite representing a small fraction of the attribute pool. The insight that VLMs learn to use contextual co-occurrence as shortcuts in few-shot settings is valuable, and the observation that removing <7% of problematic attributes yields significant generalization gains suggests the community may be over-relying on noisy attribute information. The two-stage approach of first filtering the language-side attribute pool (SAP) then addressing visual-side feature learning (SAS) provides a practical framework for the field, even if the mechanism claims are somewhat ambitious.

## Suggestions

- Reframe the SAS mechanism claims to be more precise: describe it as providing contrastive supervision through pseudo-category classification rather than claiming structural spurious-feature mitigation. The empirical results support effectiveness without requiring specific mechanism guarantees.

- Add ablation with random pseudo-categories to demonstrate that gains specifically come from spurious-attribute-targeting rather than additional training data.

- Include at least one established spurious-correlation benchmark (Waterbirds CelebA-HairColor) to report worst-group accuracy alongside the existing OOD evaluations.

## Score and Decision

**Calibration anchors used:**
- **High-scoring papers** (7–8): DiWRG9JTWZ (scores 8,5,8; MetaCoCo spurious-correlation benchmark), QuVlUn4T2G (scores 8,8,3,8; strong empirical results). These had either cleaner theoretical grounding or benchmark contributions.
- **Medium-scoring papers** (5–6): tnBaiidobu (5,6,6,6; CLIP OOD investigation, accepted), MyVC4X5B2X (5,6,6,6; debiasing with SEBRA, accepted), SksPFxRRiJ (5,5,5,5; spurious concept removal, rejected). These papers had solid contributions with some methodological limitations.
- **Low-scoring papers** (3–4): Pb9PIECnNF (3,3,5,5; unsupported claims), lEsNGN1SjG (3,3,1,1; fundamental methodological issues). These had core claim invalidation.

This paper sits between the medium and high-scoring anchors. The empirical results across 11 datasets with consistent improvements are stronger than the medium anchors, but the mechanism claims are slightly more ambitious than the evidence supports. The evaluation lacks the specialized spurious-correlation benchmarks that would anchor the claims more firmly. This positions it solidly in the upper-mid range.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>