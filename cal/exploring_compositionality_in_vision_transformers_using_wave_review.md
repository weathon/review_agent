=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary
This paper proposes a framework to test representational compositionality in Vision Transformers (ViTs) by treating Discrete Wavelet Transform (DWT) sub-bands as input-dependent primitives, drawing an analogy to the Andreas (2019) framework for NLP. Starting from the formal notion of a group homomorphism, the authors progressively relax the condition to a learned linear combination of sub-band representations and train a lightweight composition function (g*) via classification loss. Their primary finding is that for Level 1 DWT, a learned 4-weight linear combination of sub-band representations at the last encoder layer achieves accuracy (0.77–0.80) nearly on par with the original ViT (0.79–0.81), while Level 2 decomposition fails dramatically (~0.51–0.63 vs. 0.82–0.83).

---

## Strengths

- **Principled choice of DWT as primitives:** The paper gives a clear, theoretically grounded reason for preferring DWT over pixel-canonical or Fourier bases — DWT provides spatial-frequency localization and exact invertibility (Eq. 1), which makes it uniquely suitable for a compositionality study where the primitive-to-whole reconstruction is lossless by construction. This is a specific, defensible design choice, not a trivial one.

- **Honest negative reporting integrated into the framework:** The paper transparently presents the strict linearity test (Eq. 3 / Figure 2 / Figure 1) as a negative result before moving to the learned relaxation, rather than hiding the failure. This keeps the scientific chain of reasoning visible and credible.

- **Meaningful ablation of sub-band contributions (Table 5):** The comparison of LL-alone (0.494 accuracy) vs. the full learned composition (0.771) demonstrates that detail bands contribute significantly to the recovered signal — a concrete result that counters the potential trivial-baseline interpretation.

- **Multi-model and multi-wavelet coverage:** Testing across ViT-B, ViT-L, Haar, and db4 wavelets, with three constraint regimes (Convex, Conic, Unconstrained), provides consistent results that show the main finding is not an artifact of a single model or basis choice.

---

## Weaknesses

### Fatal
None. The core positive result (Level 1 compositionality under learned relaxation) is real, if narrower than claimed.

### Major

- **Unjustified conceptual drift: homomorphism → learned classification-loss regression.** The paper opens with a formal group-homomorphism definition (§2.2) and ends up training a 4-scalar linear regression minimizing classification cross-entropy (Eq. 5). Each relaxation step (strict additivity → approximate representation equality → approximate classification output) is individually plausible, but the cumulative drift is never explicitly acknowledged or justified. The abstract's claim that "primitives from a one-level DWT representation satisfy compositionality" materially overstates what is shown; what is demonstrated is that a learned linear probe over classification loss achieves near-original accuracy, which is a different (weaker) claim. This ambiguity is not resolved in the body.

- **Inconsistent "Original" accuracy across DWT levels in Table 1.** For ViT-B, the "Original" accuracy is 0.792 for Level 1 experiments but 0.83 for Level 2 experiments. For ViT-L, it is 0.809 vs. 0.82. This is the same pretrained model on a held-out test set — the original accuracy must be identical unless different subsets were used for the two experiments. No explanation is offered. If the test sets differ across levels, cross-level comparisons (which the paper relies on to characterize the Level 2 failure) are confounded.

- **No ablation baselines to rule out trivial explanations for the detail-band contribution.** Table 5 shows that LL alone yields 0.494 and the full learned composition yields 0.771, a gap the paper attributes to compositional integration of detail bands. However, no control is provided for whether this gap exists because of genuine compositionality or because any complementary partial-information signal fills in. Concretely: if the detail coefficients are replaced with (a) Gaussian noise of the same energy, (b) shuffled detail bands from randomly selected other images, or (c) a low-resolution version of the image at similar spatial bandwidth, does the gap persist? Without such controls, the 0.277 accuracy improvement is consistent with trivially combining a downsampled image with any partial signal that retains some classification-relevant structure.

- **Level 2 DWT failure is the paper's most informative result, yet it is almost entirely unexplored.** The accuracy collapse at Level 2 (from ~0.83 to ~0.51 for ViT-B) could be due to (i) ViT patch tokenization misalignment with sub-band spatial resolution at Level 2, (ii) the ViT simply not encoding deeper frequency hierarchies in a linear fashion, or (iii) catastrophic information loss in Level 2 primitives that no linear combination can recover. The paper mentions this as a limitation in one sentence of the conclusion without mechanistic analysis. This failure actually provides the strongest interpretive leverage on the paper's central question and deserves a dedicated diagnostic.

### Minor

- **Weight instability acknowledged but not analyzed.** The paper itself notes "there is no discernible pattern among the parameters. There is a lot of variation among the weights" (§4.2). Weights like [2.02, −0.18, 0.43, 0.18] vs. [0.66, 0.11, 0.10, 0.12] for different constraint regimes on the same model/dataset suggest either an ill-conditioned optimization landscape or that multiple near-equivalent solutions exist. If compositionality were a structural property, a more stable solution manifold would be expected. The paper raises the right observation but does not investigate whether this variance is random or structured (e.g., class-dependent).

- **Loss objective (Eq. 5) tests classifier behavior, not encoder compositionality directly.** The paper argues that "distance metrics in high-dimensional space are unreliable" as motivation for using classification loss, but provides no evidence for this claim and does not test whether the result changes under representation-space metrics on reduced-dimension projections (e.g., after PCA). Since g* is trained to match the *classifier head's output* (including the classifier's own errors), the positive result (Table 2's relative accuracy of 87–92%) measures how well g* imitates the ViT's decisions, not whether the encoder's internal representation is structured compositionally.

- **SSIM applied to reshaped encoder representations (§3.1).** SSIM is a perceptual metric for image quality, designed to compare 2D spatial intensity maps. Reshaping N×D encoder tokens into W×H×C for SSIM comparison involves a spatial reinterpretation that is arbitrary and potentially misleading. The paper presents this analysis as inconclusive (Figure 1), which is unsurprising given the metric choice; this analysis adds little to the paper.

- **Last-layer restriction is acknowledged but the CKA finding inverts the narrative.** Figure 2 shows CKA peaks at layers 2–4 (~0.78) before collapsing to ~0.40 at later layers. If compositionality is to be found at any layer, it is most present at early-to-middle layers, not the final one chosen for the main experiments. The paper acknowledges intermediate-layer analysis as future work, but this is notable because the positive result at the final layer may be conservative; the paper's framing should reflect this.

### Tiny

- **Section 3 framework evolution is hard to follow.** The paper moves from Eq. 3 (strict homomorphism test), to Eq. 4 (approximate representation equality), to Eq. 5 (classification loss proxy) without signposting what is sacrificed at each step. Readers must infer the motivation for each relaxation.

---

## Nice-to-Haves

- **Decomposition baselines:** Compare DWT against Fourier component splitting, random grid partitioning, and superpixel segmentation as alternative primitive-generating strategies. If g* achieves similar relative accuracy with arbitrary partitions, the case for DWT as a principled primitive basis weakens considerably.
- **Cross-architecture validation:** Testing whether a CNN backbone (e.g., ResNet) produces similar Level 1 compositionality would determine whether the finding is ViT-specific (tied to the attention mechanism and CLS token) or a generic property of deep hierarchical vision models.
- **Patch-level / spatial compositionality:** Extending analysis beyond the CLS token to patch tokens would test whether spatial compositional structure is preserved, which is arguably more relevant to ViT interpretability.
- **Attention map visualization on sub-bands:** Showing whether the model attends to semantically coherent regions in the detail bands (LH, HL, HH) would help establish whether the detail bands carry interpretable information or are acting as noise regularizers.
- **Class-conditional weight analysis:** Checking whether the learned weights η* vary systematically across semantic categories (e.g., texture-heavy vs. shape-heavy classes) would illuminate whether the composition rule is universal or class-dependent.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **No statistical significance testing (Harsh Critic §4).** Single-run accuracy evaluation without confidence intervals is the norm for ImageNet-scale experiments at ICLR. Removing this criticism.

- **Andreas (2019) analogy is strained (Harsh Critic §2–3).** The paper explicitly frames the DWT analogy as an adaptation of, not an equivalence to, the NLP setting. The paper itself acknowledges that image space is continuous (§2.2) and that the mapping is approximate. The analogy is reasonable given the framing. Removing as a standalone criticism, though the conceptual drift weakness above captures the more substantive concern.

- **No discussion of generalization to detection/segmentation (Harsh Critic §5).** This is scope creep; the paper is an exploratory interpretability study on classification representations. Removing.

- **Paper ignores CNNs, making ViT-specific framing unjustified (Harsh Critic §5).** The paper's stated contribution is specifically about ViTs. CNN comparison is a nice-to-have, not a requirement for the validity of the ViT analysis. Removing as a weakness; moved to Nice-to-Haves.

---

## Novel Insights

The most genuinely interesting observation — underplayed in the paper itself — is the CKA profile in Figure 2: compositionality w.r.t. DWT primitives is *highest at intermediate encoder layers* (2–4, CKA ~0.78) and decays sharply thereafter (~0.40 from layer 5 onwards). This pattern aligns with the known representational phase transition in ViTs (from local to global aggregation) and suggests that DWT sub-bands are most "native" to the early feature hierarchy — an insight that the paper's exclusive focus on the last layer actively obscures. Investigating whether the *positive* Level 1 result strengthens significantly at layers 2–4 and whether the Level 2 result also improves at earlier layers could substantially deepen the paper's contribution and reframe the narrative from "ViTs are approximately compositional at the final layer" to "ViT compositionality with respect to frequency primitives degrades with layer depth in a structured way."

---

## Suggestions

1. **Resolve the accuracy discrepancy in Table 1** — confirm that the same test split was used for Level 1 and Level 2 experiments; if not, re-run with a fixed held-out set and report all original accuracies as identical.
2. **Add two control baselines to the detail-band ablation** — replace detail coefficients with (a) random Gaussian noise matched in energy and (b) detail bands from randomly selected images not in the test set, and report resulting accuracies alongside Table 5.
3. **Restate the abstract claim** — replace "satisfy compositionality" with "exhibit approximate compositionality under a learned linear probe," to accurately characterize the experimental evidence.
4. **Dedicate a full subsection to Level 2 failure diagnosis** — ablate sub-band spatial resolution vs. ViT patch size (16×16 patches become 8×8 at Level 2), and test whether a ViT with smaller patch size (e.g., ViT with 8×8 patches) recovers Level 2 compositionality.
5. **Extend the intermediate-layer analysis** — report the g* accuracy for layers 2–4 (where CKA peaks) alongside the final-layer result to give a complete picture of how compositionality varies with encoder depth.
6. **Clarify the framework evolution in §3** — add a brief paragraph before Eq. 5 that explicitly enumerates the three relaxations made from the original homomorphism definition and states what interpretive consequence each relaxation has.

---

**Overall assessment:** The paper raises a genuinely interesting research question and makes a creative connection between signal processing and ViT interpretability. The Level 1 compositionality result is real and reproducible across models and wavelet bases. However, the paper is currently at a preliminary stage: the central conceptual drift is insufficiently acknowledged, critical ablation controls are absent, the paper's most informative result (Level 2 failure) is left mechanistically unexplained, and the accuracy inconsistencies raise methodological concerns. **Novelty** is moderate — the DWT-as-primitive idea is creative but narrowly applied. **Technical soundness** is weak-to-moderate due to the unacknowledged conceptual gap and missing controls. **Empirical support** is limited. **Significance** is limited in its current form; the work reads as a pilot study. **Clarity** needs improvement in the framework derivation. In its current state, the paper does not meet ICLR's bar, but could be substantially strengthened with the targeted revisions identified above.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 5.0]
Average score: 5.3
Binary outcome: Reject
