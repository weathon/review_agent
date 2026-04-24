## Summary

This paper proposes Language-Assisted Feature Transformation (LAFT), a training-free method that manipulates CLIP image embeddings using text-derived concept axes to enable user-guided anomaly detection. The method supports both "guiding" (projecting onto relevant attribute subspaces) and "ignoring" (projecting orthogonal to nuisance attribute subspaces), and is evaluated on semantic anomaly detection (Colored MNIST, Waterbirds, CelebA) and industrial anomaly detection (MVTec AD, VisA).

## Strengths

- **Novel training-free concept subspace construction.** The paper introduces a distinct mechanism for feature transformation that differs from prior VLM-based anomaly detection methods requiring adapter fine-tuning (e.g., InCTRL, APRIL-GAN). It constructs concept axes by applying PCA to pairwise differences between CLIP text embeddings of concrete attribute values (Section 4.2), enabling training-free manipulation of visual features.
- **Strong semantic anomaly detection results.** LAFT AD achieves substantially higher AUROC than language-only and few-shot baselines on semantic benchmarks. Table 1 shows 98.5% AUROC on Colored MNIST (vs. 94.0% for InCTRL) and 95.6% on Waterbirds (vs. 92.2% for ZOE/WinCLIP). Table 2 shows 98.1% AUROC on CelebA Eyeglasses (vs. 87.8% for InCTRL).
- **Simplicity and deployability.** LAFT requires only forward passes through a frozen CLIP model and is agnostic to the downstream anomaly detector, making it lightweight and practical.
- **Explicit language-driven suppression of nuisance attributes.** Unlike prior work that requires labeled nuisance attributes, LAFT supports both guiding and ignoring attributes through orthogonal projection (Section 4.3). Table 1 shows 97.4% AUROC on Colored MNIST when suppressing color.

## Weaknesses

### Fatal
None.

### Major
- **Missing empirical validation of the core "ignore" mechanism.** Section 3 promises that "invariance can be measured by the accuracy of predicting the anomaly label ... [and] informativeness by measuring the accuracy of predicting the relevant attribute," and explicitly states that "empirical evaluations of these measures for various datasets are provided in the Experiments." However, the experiments (Tables 1–4) report only anomaly detection AUROC/AUPRC/FPR95, with no direct analysis of whether irrelevant attributes are actually removed from transformed features (e.g., by training a classifier to predict the ignored attribute from post-projection features). Without this validation, the central claim that LAFT can "selectively focus on or ignore specific image attributes" relies on anomaly detection performance as indirect evidence, which is insufficient to establish that the projection truly achieves the invariance desideratum in Eq. (1).

### Minor
- **Industrial AD evaluation conflates fair and privileged-knowledge settings.** The paper presents both LAFT-G (general prompts using "state words and category names") and LAFT-C (category-specific prompts using anomaly class names like "bottle with large breakage"). While LAFT-G already outperforms WinCLIP+ without using anomaly names (92.6 vs. 90.4 AUROC zero-shot on MVTec AD, Table 3), the paper does not clearly frame LAFT-C as an upper-bound or proof-of-concept rather than a fair comparison. Because zero-shot methods like WinCLIP do not assume foreknowledge of specific defect types, presenting LAFT-C alongside baselines without explicit qualification risks misleading readers about what knowledge is required.
- **Unjustified geometric assumptions.** The paper asserts that "irrelevant attributes are nearly orthogonal to the concept axes" (Section 4.3) and that PCA on pairwise text embedding differences yields axes aligned with intended semantic attributes rather than template noise (Section 4.2). These are strong assumptions about CLIP's embedding space that are neither theoretically justified nor empirically verified in the main text.
- **Limited reproducibility detail for WinCLIP+LAFT in the main paper.** The industrial extension is described in a single paragraph (Section 5.2) stating that LAFT is applied to "window, image, and text embeddings," but the main text lacks algorithmic detail, figures, or dimensional analysis for how projection interacts with WinCLIP’s multi-scale patch features. Greater clarity in the main paper would improve reproducibility.

### Trivial
- The gap between Eqs. (1)–(2) (stated desiderata) and the projection heuristic (Section 4.3) is not acknowledged as a heuristic approximation.

## Nice-to-Have
- Disentanglement analysis quantifying how much variance of the irrelevant attribute remains in the projected subspace would strengthen the paper's mechanistic claims.
- Qualitative failure cases showing where LAFT fails to ignore an attribute would honestly characterize the limits of linear projection in CLIP space.
- Industrial AD qualitative results (e.g., WinCLIP vs. WinCLIP+LAFT heatmaps) would help interpret the AUROC gains in Table 3.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"Anomalous baseline results on Waterbirds undermine experimental credibility."** The critic finds it suspicious that LAFT AD (95.6 AUROC) outperforms LinearProbe (91.0 AUROC) on Waterbirds. However, the paper explicitly states that LinearProbe is "a linear classifier to predict the class of the test image," not an anomaly detector. In the Waterbirds setup—where the training set has a strong spurious correlation between bird type and background—a classifier trained on all data learns this correlation and is not optimized for anomaly detection. The underperformance is expected, not suspicious. Similarly, the small gap between kNN with subset (82.3) and all normal images (83.0) is consistent with the paper's thesis: simply adding more correlated data does not break spurious correlations, whereas LAFT explicitly projects onto a bird-type concept axis.
- **"Table 4 contradicts the paper's premise."** The inclusion of an "Exact anomalies" condition in the prompt ablation is not a contradiction; it is a standard ablation study showing performance under varying degrees of prior knowledge. The paper explicitly frames Table 4 as investigating "how the performance of LAFT changes depending on the quality of the concept values provided."
- **"No algorithmic detail for WinCLIP+LAFT."** Per the review instructions, appendix sections are stripped from all papers in this format and exist in the original submission. Criticisms about absent appendix content are parser errors, not author errors. However, the main paper's presentation of this extension remains thin even accounting for this.

## Novel Insights

The paper's core insight—that pairwise differences between CLIP text embeddings of concrete attribute values can define concept axes for feature transformation—is genuinely novel and practically appealing. The Colored MNIST visualization (Figure 3) effectively communicates this intuition. However, the paper would be significantly strengthened by directly validating that orthogonal projection off these axes actually removes the targeted attribute from the representation, rather than relying solely on downstream anomaly detection accuracy as a proxy.

## Suggestions

1. Add direct empirical validation of the ignore mechanism by training classifiers to predict ignored/relevant attributes from transformed features, as promised in Section 3.
2. Clearly distinguish in the industrial AD section that LAFT-G is the fair comparison to WinCLIP/ZOE, while LAFT-C demonstrates an upper bound with additional category-specific knowledge.
3. Provide empirical analysis (e.g., cosine similarities between concept axes and known attribute directions) to support the orthogonality assumption.

## Score and Decision

**Calibration:** I compared this paper against several anchors. High-scoring anchors include AnomalyCLIP (6.17, Accept poster) and One-for-All Few-Shot AD (6.40, Accept poster), which share the VLM-based anomaly detection topic and have comprehensive experiments but minor weaknesses. Medium-scoring anchors include Graph AD via forward diffusion (5.17, Reject) and Unified Uncertainty Estimation (5.25, Reject), which have novel elements but gaps in validation or theory. Low-scoring anchors include Variance-Stabilized Density Estimation (3.00, Withdrawn), which has fundamental methodological issues and critical missing ablations.

LAFT has a more novel and elegant core idea than the medium anchors, and its semantic AD results are strong. However, it shares with the medium anchors a significant gap in empirical validation: the paper promises direct validation of attribute removal but does not deliver it. The industrial AD presentation issue is less severe than the unfair-comparison concerns in the medium anchors because LAFT-G provides a fair baseline. Overall, this paper sits slightly above the medium band but below the established high-scoring VLM-AD papers due to its narrower validation and thinner industrial AD exposition.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>