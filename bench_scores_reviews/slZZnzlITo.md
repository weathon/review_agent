## Summary
This paper proposes Multimodal Open Set Recognition (MMOSR), extending OSR to multimodal data, and empirically diagnoses "fusion degradation"—the failure mode whereby naïvely applying OSR regularization to fused multimodal representations over-compresses the feature space and degrades both closed-set accuracy and unknown detection. To address this, the authors introduce the Multimodal Representation Reactivation Network (MRN), which combines bidirectional cross-attention (Mutually Enhanced Fusion) with a Mixture-of-Experts classifier (Adaptive Fusion). Experiments across four datasets covering image-text, audio-visual, and RGB-depth modalities show consistent, if sometimes modest, improvements over both unimodal OSR and multimodal fusion baselines.

---

## Strengths

- **Clean identification of a concrete failure mode.** Table 1 on Food-101 delivers a targeted diagnostic: Text-OSR achieves AUROC ≈ 90–92, Fusion improves closed-set ACC at a small AUROC cost, but Fusion-OSR drops both AUROC and ACC relative to Fusion alone (e.g., −2.25 AUROC / −5.23 ACC at the 10-class split). This controlled ablation—isolating modality, fusion, and OSR independently before combining them—is more careful than a typical systems paper and provides a useful warning to the community against naïve combination.

- **Dual-role design: standalone method and plug-in fusion module.** MRN is evaluated not only as a standalone MMOSR model but as a drop-in fusion backbone inside ARPL and CSRR (ARPL-MRN, CSRR-MRN). In both roles it consistently outperforms ADD/CAT/GQA fusion counterparts across all four datasets and both OSR loss families, demonstrating that the architecture is a genuinely reusable component rather than a one-shot solution.

- **Consistent positive gains as a plug-in.** In the plug-in setting (Table 2), MRN yields positive gains over the next-best fusion strategy in every single cell across both OSR methods and all four datasets. This consistency is meaningful, even if the margins are sometimes small.

- **Breadth of benchmark.** Four datasets spanning three heterogeneous modality pairs (image-text, audio-visual, RGB-depth) and varying numbers of known/unknown classes (5–80 in Table 3) provide meaningful diversity. All baselines are reproduced by the authors using the same encoder, ensuring fairer comparison than re-using published numbers.

---

## Weaknesses

### Fatal
None.

### Major

- **MoE ablation is entirely missing.** Table 4 ablates only the two cross-attention branches (C₁, C₂) while the Adaptive Fusion / MoE module is always present in every row. There is no experiment comparing MoE against a single MLP of matched capacity. Because MRN attributes part of its gain to "adaptive fusion capturing multiple informative representations," the absence of this ablation leaves one of the two central architectural claims without direct support. The observed gains could plausibly arise entirely from cross-attention, with MoE contributing nothing beyond a larger parameter count.

- **Fusion degradation diagnosis is limited to a single dataset.** The entire Section 3.2 motivation—including Table 1 and Figure 2—is built on Food-101 alone with a single simple addition-based Fusion-OSR baseline. There is no analogous diagnostic table or visualization for Flower-102, CREMA-D, or SUN RGB-D. Given that CREMA-D and SUN RGB-D have very different modality characteristics, the generality of "fusion degradation" as a universal MMOSR pathology remains unverified. This weakens the necessity argument for a new task and the targeted motivation for MRN's design.

- **No variance estimates; reported gains are often within plausible noise.** All results are single-point numbers with no standard deviations, confidence intervals, or multiple-seed averaging. In Table 2, several improvements are extremely small: SUN RGB-D +0.37 AUROC / +0.01 OSCR over the next-best fusion method, and CREMA-D is actually negative (−1.05 AUROC / −0.18 OSCR vs. MLA). Sub-1% improvements are uninformative without variance. For a paper that defines a new benchmark task, this is a significant reproducibility and credibility concern.

### Minor

- **MRN underperforms MLA on CREMA-D, yet the paper claims consistent superiority.** Table 2 shows MLA beats MRN on CREMA-D (67.83 vs. 66.78 AUROC; 57.50 vs. 57.32 OSCR). The paper's narrative that MRN "consistently demonstrates exceptional MMOSR performance" does not acknowledge this exception. The paper should be candid about where improvements hold and where they do not, and analyze why CREMA-D behaves differently.

- **Unknown rejection mechanism is entirely standard.** Section 4.3 uses maximum Softmax probability thresholded to retain 95% of known samples—a baseline known to be suboptimal in OSR/OOD literature (energy scoring, Mahalanobis distance, OpenMax, etc. all outperform it in many settings). No comparison is made against alternative scoring functions applied to the same MRN features. It is therefore unclear how much of any gain comes from the learned representation versus the scoring function, and whether a stronger scorer atop a weaker backbone could match MRN.

- **Fusion degradation is measured only qualitatively.** The claim that OSR regularization "over-compresses" fused representations is supported exclusively by t-SNE visualizations and downstream metric drops. t-SNE is a nonlinear embedding that does not reliably preserve global structure; it is not a measurement of representation compactness, rank, or discriminability. Quantitative diagnostics (e.g., feature covariance rank, effective dimensionality, CKA similarity between fused and unimodal representations with/without OSR loss) are needed to substantiate the stated mechanism.

- **OSCR metric is never formally defined.** Despite being used in all main tables and highlighted in the headline claim ("up to 5.23% on OSCR"), the OSCR formula does not appear anywhere in the paper's main text. Readers unfamiliar with the metric cannot verify whether the evaluation protocol matches the original definition.

- **Benchmark construction protocol underspecified.** The text does not state how many random seeds are averaged per result, how known/unknown class splits are drawn (fixed seed vs. average over random splits), or how the threshold τ is calibrated in practice (validation split? training split?). These details are necessary to reproduce any result and to assess whether the gains are stable across splits.

### Tiny

- **Equation (1) has ambiguous notation.** The expression `Softmax(W₁^Q z₁ z₂ W₁^K / √d)(W₁^V z₂)` is dimensionally unclear without explicit transposes and shape annotations. Standard cross-attention would require `z₂ᵀ` to form the inner product. The missing transpose makes it impossible to verify the operation from the equation alone.

- **Metric inconsistency across sections.** Table 1 / Table 4 use AUROC + ACC; Table 2 uses AUROC + OSCR; Table 3 uses both. If OSCR is the preferred MMOSR metric (it captures both open-set and closed-set performance jointly), it is odd that it is absent from the key necessity analysis (Table 1) and ablation (Table 4).

---

## Nice-to-Haves

- **Computational cost analysis.** Adding a table comparing parameter count, FLOPs, and inference latency between MRN and simpler fusion baselines (ADD, CAT, MLA) would help practitioners evaluate the performance-cost trade-off, especially for the robotic deployment scenarios mentioned in the introduction.

- **Fusion degradation visualization on other datasets.** A Figure-2-style t-SNE (or, better, quantitative analysis) for CREMA-D and SUN RGB-D would substantiate whether the phenomenon is general. Even a brief note explaining why CREMA-D behaves differently from Food-101 would be valuable.

- **Missing-modality robustness.** The paper's practical motivation (robots, unmanned systems) strongly implies sensors can fail. Testing MRN under one-modality-missing at inference time—and comparing it to unimodal fallback—would directly address the practical scenario and could reveal a natural advantage of the bidirectional cross-attention design (graceful degradation when one branch has no query signal).

- **Stronger fusion+OSR baselines.** The "Multimodal fusion with OSR" rows in Table 2 use ADD/CAT/GQA as the fusion component. Using TMC or MLA (the strongest fusion baselines) as the backbone for ARPL/CSRR would be a fairer stress test of MRN as a plug-in and would tighten the experimental claims.

- **Alternative OSR scoring functions.** Comparing energy scoring, Mahalanobis distance, or OpenMax on top of MRN features would disentangle representation quality from threshold selection and likely improve reported AUROC further.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Figure 2 (d) description contradicts the paper's narrative" (Harsh Critic).** The critic claims Figure 2d (Fusion-OSR) is described as having "dispersed unknowns," which would contradict the paper's narrative of over-compression. The alt-text used by the critic is an AI-generated image description and is likely unreliable. The quantitative evidence in Table 1 clearly and consistently shows Fusion-OSR degrades both AUROC and ACC relative to plain Fusion, supporting the paper's narrative. This criticism should not be trusted.

- **"GQA is not a canonical multimodal fusion baseline" (Harsh Critic).** GQA (Grouped Query Attention, EMNLP'23) is used as a fusion mechanism, not in its NLP efficiency role. Its use is unusual but not unreasonable as an attention-based fusion baseline. Per the rules, citing a published method is sufficient to assume it exists and can be used as a baseline.

- **"CLIP comparison is unfair" (Harsh Critic).** CLIP operates in a zero-shot or 16-shot fine-tune regime while MRN trains from scratch on known classes. The comparison is acknowledged to be across different data regimes and is presented as an additional reference point, not as the primary apples-to-apples comparison. The paper does not misrepresent the comparison. This is not a fairness problem—if anything, CLIP has a massive pretraining advantage, so MRN beating it strengthens rather than weakens the paper's claim.

- **"Lack of theoretical grounding for fusion degradation" (multiple reviewers).** This paper is an empirical systems contribution. Demanding theoretical proofs of a diagnosed failure mode is not a standard expectation for this type of work. Weakened to the quantitative diagnostic request already listed under Weaknesses/Minor.

- **"Missing related works on multimodal OOD / uncertainty estimation" (Harsh Critic).** Per instructions, potential missing related works are not included as we cannot verify their existence.

- **"The problem definition is too conventional to be a contribution" (Harsh Critic).** The task formulation is intentionally minimal; the contribution lies in the empirical diagnosis and the proposed method. Evaluating a paper negatively for scoping its formal definition narrowly is unfair.

---

## Novel Insights

The most genuinely useful insight this paper surfaces—and one that practitioners should take seriously—is that OSR regularization applied naïvely to multimodal fusion is not a "free lunch." Because OSR losses (e.g., ARPL's reciprocal-point loss) penalize feature spread to compress known-class representations, they work against the goal of multimodal fusion, which needs diverse, modality-specific information to be preserved. The result is that adding an OSR regularizer to a strong multimodal backbone can *hurt* both closed-set accuracy and open-set detection simultaneously, even though each component works well in isolation. This failure mode is subtle because it is non-obvious that the same inductive bias that helps in unimodal OSR (compactness of known classes) is harmful when applied to fused representations that need to retain cross-modal diversity. The design response—using cross-attention to force modality representations to actively inform each other before compression, and MoE to preserve representation diversity after—is a principled architectural answer to this diagnosed pathology, even if it draws on standard components.

---

## Suggestions

1. **Add MoE ablation.** Add a row to Table 4: "C₁ ✓, C₂ ✓, single MLP (same capacity)" vs. "C₁ ✓, C₂ ✓, MoE (E=15, K=4)" on both Food-101 and Flower-102. This is the single most important missing experiment and can be done with existing infrastructure.

2. **Report mean ± std over ≥ 3 random class splits** for all main results in Table 2 and Table 3. Given that some improvements are below 0.5%, single-run results are insufficient to substantiate the claims.

3. **Extend the fusion degradation diagnostic (Table 1 equivalent) to at least one additional dataset** (CREMA-D or SUN RGB-D), using the same Image-OSR / Text-OSR / Fusion / Fusion-OSR structure. This directly validates the generality of the phenomenon that motivates the paper.

4. **Formally define OSCR in the paper** (e.g., in Section 5.1 or an appendix), and ensure Table 4's ablation uses it so that all results are interpretable on the same evaluation axis.

5. **Add a quantitative representation-compactness measurement** (e.g., effective rank of the covariance matrix of class embeddings, or average pairwise cosine distance among class centroids) measured before and after applying OSR regularization to fusion representations. Even one such figure would transform the "fusion degradation" claim from qualitative to quantitative.

6. **Explicitly address the CREMA-D exception** in the results discussion: explain whether the underperformance relative to MLA is expected (e.g., audio-visual fusion behaves differently), or identify what aspect of CREMA-D makes it structurally different from the other datasets.

---

## Evaluation

**Originality:** Moderate. The task definition is a natural extension of OSR to multimodal settings—not a conceptual surprise—but the concrete diagnosis of fusion degradation and the targeted architectural response are genuine contributions beyond prior work. The individual components (cross-attention, MoE) are standard; their combination in this specific context is new and well-motivated.

**Importance of research question:** High. Open-set recognition in multimodal systems is a practically significant and underexplored problem. The finding that naïve fusion+OSR combinations can *degrade* performance is a practically important warning for system builders.

**Claims well-supported:** Partially. The existence of fusion degradation is convincingly shown on Food-101 but not generalized. MRN's effectiveness is supported in 3/4 standalone comparisons and all plug-in comparisons, but the absence of variance estimates and the incomplete ablation weaken confidence in the magnitude of the gains.

**Soundness of experiments:** Moderate. The experimental design (same encoders across methods, training from scratch on known classes only, diverse modality pairs) is sound in principle, but missing variance, an incomplete ablation, and a single-dataset motivation analysis are meaningful gaps.

**Clarity of writing:** Generally clear with a logical flow from problem to diagnosis to method. The equation notation in (1) is ambiguous and the OSCR metric is never defined, but these are addressable.

**Value to the research community:** Moderate to high. The paper creates a useful benchmark and identifies a reproducible failure mode that is likely to be of interest to anyone combining OSR with multimodal architectures.

**Contextualization relative to prior work:** Adequate for OSR and multimodal fusion. The connection to the broader multimodal uncertainty and OOD detection literatures could be developed further to sharpen the novelty claim, though this is not a blocking issue.