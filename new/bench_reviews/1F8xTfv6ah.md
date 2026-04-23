Now let me run calibration searches in parallel:Now I have enough to write the final consolidated review.

---

## Summary

This paper proposes an OOD detection method that exploits the local neuroplasticity property of Kolmogorov-Arnold Networks (KANs): by comparing activation patterns of a trained KAN against an identically initialized untrained KAN, test samples that activate regions not trained on InD data are flagged as OOD (formalized in Eq. 5). Because raw KANs only capture marginal feature distributions, the authors introduce a class-label-based dataset partitioning scheme (Section 2.3) to approximate joint density. The method is evaluated on seven benchmarks spanning image (CIFAR-10/100, ImageNet-200/1K full-spectrum) and tabular medical domains.

---

## Strengths

- **Strong and consistent empirical wins on the ImageNet benchmarks (Table 2):** KAN achieves 71.46 vs. 67.18 (ASH) on ImageNet-200 FS, and 78.52 vs. 76.28 (NAC) on ImageNet-1K FS — meaningful margins on non-trivial large-scale benchmarks where evaluation is competitive and near-OOD is hard.

- **Exceptional dataset-size robustness (Table 6):** AUROC remains stable from 100% down to 0.1% of CIFAR-10 training data (94.12 → 93.21), while KNN collapses to 8.15 and VIM to 76.38 at 0.1%. This is a practically significant and empirically well-supported advantage, particularly relevant for data-scarce deployments.

- **Interpretable, principled formulation (Eq. 5 and Figure 2):** The decomposition of the Δ matrix into "where InD information is stored" × "where a test sample activates" is a clean and unusually interpretable detection criterion, clearly illustrated by the toy example.

- **Honest spline-vs-histogram ablation (Section 3.3):** The histogram baseline (85.29%) vs. KAN detector (94.12%) isolates the value of spline smoothing over binary bin counting — a real and clearly presented ablation.

- **Cross-domain validation:** Results on FT-Transformer-based tabular medical benchmarks (Tables 3–5) demonstrate the method is not tied to vision backbone architectures.

- **Task-agnostic training (Section 3.3):** The finding that regression to a constant yields ≈0.2% AUROC difference on image benchmarks supports the claim that the mechanism is truly about registering InD samples in spline coefficients rather than relying on task-specific supervision.

---

## Weaknesses

### Fatal
None.

### Major

- **Overclaiming on the CIFAR-100 benchmark.** The abstract and Section 3.2 state the method "outperforms all previous methods on both benchmarks." On CIFAR-100, KAN achieves Avg Overall AUROC of **83.44 ± 1.99** vs. NAC's **83.36 ± 0.84** — a 0.08-point gap with substantially overlapping error intervals (KAN's variance is 2.4× larger). More importantly, KAN's near-OOD AUROC on CIFAR-100 is **77.17**, substantially below RMDS (80.15), GEN (81.31), KNN (80.18), and ReAct (80.52) — all methods the claim subsumes. The headline is defensible for CIFAR-10 and ImageNet but misleading for CIFAR-100, and the paper's own statistical bolding methodology should have flagged this.

- **The contribution of the KAN architecture per se is not isolated from class-conditional partitioning.** The method combines at least three design choices: (1) class-conditional partitioning (the primary driver, as Table 7 shows P=1 → P=10 jumps AUROC from 46% to 94%), (2) multi-layer backbone feature fusion (explicitly credited to NAC, Liu et al. 2024a), and (3) KAN spline smoothing. The only ablation separating the KAN from a simpler model is the histogram comparison (Section 3.3), but this does not control for (1) and (2) simultaneously. The natural direct baseline — a class-conditional Gaussian or GMM trained on the same partitioned, multi-layer backbone features — is absent. Without it, the performance advantage over NAC cannot confidently be attributed to the KAN spline architecture rather than the partitioning strategy or feature representation.

### Minor

- **The "local neuroplasticity" narrative is somewhat misleading in the abstract and introduction.** While the paper is transparent about the marginal-distribution limitation (Section 2.3), the abstract and introduction present local neuroplasticity as the operative mechanism. In practice, P=1 (the pure KAN leveraging only local neuroplasticity) yields 46.08 ± 15.58% AUROC — below chance — and competitive performance emerges only from class-conditional partitioning. The paper would be more precise if it framed the contribution as "class-conditional spline density estimation enabled by KAN architecture" rather than leading with local neuroplasticity as the primary driver.

- **Table 4 (Age benchmark) deserves analysis.** All methods including KAN achieve AUROC ≈ 50 (random chance), yet the paper does not discuss this. Understanding whether this reflects an inherently ill-posed task, a backbone that doesn't encode age-related shifts, or a limitation of post-hoc feature-space methods would substantially clarify the method's scope.

- **Computational cost of partitioned inference is not reported.** For CIFAR-100 with P=100 and ImageNet-1K potentially with P=1000 class-partitioned KANs, the inference overhead could be orders of magnitude higher than post-hoc methods like KNN or RMDS. This is relevant for any deployment comparison.

### Trivial

- Table 6 reports CIFAR-100 only down to 1% (not 0.1% as for CIFAR-10); a consistent comparison across the same reduction levels would strengthen the table.

---

## Nice-to-Haves

- **Class-conditional GMM/KDE baseline on identical features**: Training a Gaussian mixture model on the same class-partitioned, multi-layer backbone features would directly test whether the KAN architecture adds anything beyond the class-conditioning structure.
- **Ablation isolating multi-layer feature fusion**: Testing the KAN detector with only the last backbone layer (matching other post-hoc methods) would reveal how much of the improvement is attributable to feature engineering borrowed from NAC vs. the KAN scoring.
- **High-dimensional projection of Figure 2**: Showing backbone feature regions registered as InD by the trained KAN (with OOD samples overlaid) in a 2D PCA/t-SNE projection would validate that the mechanism generalizes beyond the 1D toy example.
- **K-means partitioning comparison at identical P for CIFAR-10**: Since CIFAR-10 uses P=10 = number of classes, it is unclear whether class structure is required or whether arbitrary geometric partitioning suffices. Testing k-means with P=10 (no labels) vs. class-label splitting would clarify this.

---

## Removed Points

*These points were raised by reviewers but removed from the main assessment for the following reasons. Treat with caution.*

- **"P=1 refutes the central theoretical claim" (called "structural"/fatal by harsh critic).** The paper is explicit and honest about this limitation in Section 2.3: "KANs constrain its ability to model the joint distribution" is directly stated, and the partitioning fix is principled and well-motivated with a clear toy demonstration. This is not a hidden flaw — the paper fully acknowledges and addresses it. What remains as a legitimate concern (framing accuracy) is captured in the Minor section above, not as a Fatal weakness.

- **Demand for a k-means vs. class-label partition ablation (P = k-means vs. P = class labels).** The paper notes in Appendix A.7 that the choice of clustering algorithm does not significantly affect detection performance. Requiring a further ablation on this is a nice-to-have but not a core methodological gap.

- **B-spline support width technicality.** The harsh critic notes that B-splines of order k=3 span k+1=4 grid intervals, making locality "less sharp than advertised." This is correct but pedantic — the local plasticity property is qualitatively real and well-documented in the KAN literature. It does not affect the results.

- **Claim that method "cannot be independently verified" or availability concerns.** Not raised explicitly here, but any reproducibility concern premised on doubting NAC or OpenOOD existence is disregarded per hard rules.

- **Missing class-conditional density methods in related work.** While the Mahalanobis distance (MDS) method does model class-conditional Gaussians, the paper does include MDS as a baseline. The absence of a dedicated related-work subsection comparing to density-based methods specifically is noted but not a serious flaw given MDS is in all result tables.

---

## Novel Insights

The most genuinely novel insight in this work — not widely appreciated in prior OOD literature — is the extreme robustness to training dataset size (Table 6). While distance-based methods like KNN require dense feature stores to reliably cover the InD manifold (collapsing to 8.15% AUROC at 0.1% of CIFAR-10), the KAN detector updates only local spline coefficients near each training point: even a handful of examples per class is sufficient to register approximate InD regions, making the method near-invariant to dataset size. This is mechanistically distinct from other post-hoc methods and practically significant for data-scarce domains. The combination of this property with the post-hoc, backbone-agnostic design makes the method particularly suitable for medical and industrial settings where labeled InD data may be scarce.

---

## Suggestions

1. Revise the abstract's "superior performance" claim to accurately reflect the CIFAR-100 result as a near-tie with NAC, while emphasizing the genuine ImageNet and dataset-size advantages.
2. Add a class-conditional GMM/MoG baseline on the same partitioned, multi-layer features to isolate the spline architecture's contribution.
3. Report inference time as a function of P to allow fair comparison with lighter post-hoc methods at scale.
4. Discuss the Age benchmark near-random result in the body of the paper — it genuinely informs the scope of applicability.
5. Reframe the introduction to present the method as "class-conditional spline-based InD region registration" rather than leading solely with local neuroplasticity, which missets expectations about what component does the heavy lifting.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Human Score | Comparison to KAN paper |
|---|---|---|---|
| HAct (activation histograms for OOD) | `Oo5spZRpH6.md` | 3.67 (Reject) | Similar concept but had credibility concerns about inflated results and limited/noisy evaluation. KAN paper is substantially more rigorous and broader. |
| MAGDiff (activation graphs for shift detection) | `l18hiEXRJS.md` | 4.50 (Reject) | Similar structural idea (comparing activations), rejected for limited baselines and datasets. KAN paper has much broader evaluation (7 benchmarks, 2 domains). |
| SCALE (post-hoc OOD via activation shaping) | `RDSTjtnqCg.md` | 6.25 (Accept poster) | Comparable post-hoc OOD paper with clear mechanistic analysis. Accepted despite limited novelty over ASH. KAN paper has more novel idea but weaker attribution of the key component. |
| ImOOD (OOD on imbalanced data) | `am7BPV3Cwo.md` | 5.75 (Reject) | Clear theoretical framework + strong results but limited scope and missing ablations. Structurally comparable to KAN paper. |
| NegLabel (vision-language OOD) | `xUO1HXz4an.md` | 7.50 (Accept Spotlight) | High bar — uses external VLM labels, much stronger novelty and experimental rigor. KAN paper is below this level. |

**Positioning:** The KAN paper sits clearly above HAct (3.67) and MAGDiff (4.50) in evaluation rigor and empirical breadth. It is comparable to SCALE (6.25) in scope, with stronger empirical results (especially ImageNet) but weaker theoretical attribution (the overclaiming issue and missing class-conditional density baseline are more significant than SCALE's weaknesses). The ImOOD anchor (5.75, rejected) is a useful parallel: similar structure, rejected despite real contributions. The KAN paper's ImageNet wins are more convincing than most medium-tier anchors, which slightly nudges the score upward — but the CIFAR-100 overclaiming and missing ablation prevent it from reaching SCALE's level.

**Final assessment:** The paper makes a genuine and practically useful contribution — the KAN-based OOD detector achieves real improvements on ImageNet-scale benchmarks and exhibits novel dataset-size robustness. However, the core claim narrative overstates what is demonstrated: the operative mechanism is class-conditional density estimation enabled by partitioning, not local neuroplasticity per se, and the performance advantage cannot be attributed to the KAN architecture without a class-conditional density baseline comparison. CIFAR-100 is overclaimed. These are correctable but non-trivial.

**Score: 5.5** — borderline, leaning toward weak reject, positioned between MAGDiff/ImOOD (4.5–5.75, rejected) and SCALE (6.25, accepted). Revisions addressing the CIFAR-100 overclaim and adding the class-conditional GMM baseline would likely push this to a clear accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>