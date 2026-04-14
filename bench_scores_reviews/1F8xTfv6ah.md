## Summary

This paper proposes a post-hoc OOD detection method that exploits the local neuroplasticity of Kolmogorov-Arnold Networks (KANs). The core idea is to train a KAN on in-distribution (InD) data and compare its activation responses to an identically initialized but untrained counterpart: InD samples strongly activate regions whose spline coefficients were updated during training, while OOD samples activate untouched regions, yielding a low difference signal. To overcome KANs' inherent limitation of modeling only marginal (not joint) feature distributions, the authors introduce a class-conditional (or k-means-based) dataset partitioning scheme that trains an ensemble of KAN detectors. The method is evaluated on seven benchmarks across image (CIFAR-10/100, ImageNet-200 FS, ImageNet-1K FS) and medical tabular domains, claiming state-of-the-art average AUROC and notable robustness to training data scarcity.

---

## Strengths

- **Exceptional data-size robustness (Table 6):** KAN maintains 93.21 ± 0.53 AUROC on CIFAR-10 at 0.1% dataset size (≈5 samples/class), while KNN collapses to 8.15 ± 0.86 and VIM degrades to 76.38 ± 3.83. This is a quantitatively striking property that most OOD methods cannot claim and is directly relevant to data-scarce industrial deployment.

- **Strong and clear gains on ImageNet-scale benchmarks (Table 2):** On ImageNet-200 FS, KAN achieves 71.46 ± 0.40 Avg Overall versus the next best (Gram: 63.65 ± 0.61), a ~8% absolute gap. On ImageNet-1K FS, KAN scores 78.52 versus NAC's 76.28. Gains on iNaturalist (84.13 vs. 65.83 for NAC) and Textures (83.30 vs. 74.41 for NAC) are particularly large and unlikely to be noise.

- **Mechanistically interpretable score decomposition (Eq. 5):** The rewriting of Δ_{p,q}(x_p) as a product of coefficient-difference terms ("where InD was stored") and B-spline basis masks ("what the test sample activates") is mathematically clean and provides genuine interpretability rare in post-hoc OOD methods.

- **Honest identification and quantified fix for the marginal distribution limitation (Section 2.3, Table 7):** The authors explicitly identify that KANs process features independently (marginal, not joint distributions), and the proposed partitioning strategy recovers joint information. The P=1 vs. P=10 ablation (46% vs. 94% AUROC) quantitatively demonstrates why the fix is necessary.

- **Spline smoothing provides real value over binary density estimation (Section 3.3):** The histogram baseline comparison — replacing splines with binary bins — achieves 85.29% vs. KAN's 94.12%, a ~9% gap. This isolates genuine value attributable to spline continuity over naive density counting.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of computational cost analysis** — The method requires training P KAN models and running P+1 forward passes at inference time (one per partition, plus the shared untrained KAN). For CIFAR-10 with P=10 this is a 10× overhead vs. a single forward pass. No training time, inference latency, memory footprint, or FLOPs comparison with baselines is provided anywhere in the paper. For a method positioned for real-world/industrial deployment (Mercedes-Benz affiliation), this omission is surprising and substantially limits the reader's ability to assess practical viability.

- **Multi-layer feature fusion fairness unconfirmed** — The paper uses multi-layer backbone features, following Liu et al. (2024a)/NAC, which also uses multi-layer features. However, Section 3.1 does not state whether the other baselines (KNN, VIM, RMDS, MDS, etc.) are evaluated with multi-layer features or only the final backbone layer. If only the KAN detector and NAC use multi-layer features while others use single-layer, the comparison is confounded — the performance gap over KNN/VIM/RMDS may partly reflect the richer feature representation rather than the KAN mechanism. This fairness question must be explicitly addressed.

- **Histogram normalization not ablated** — Section 3.1 describes histogram normalization as a critical preprocessing step that maps the highly skewed backbone feature distribution onto the KAN grid's uniform range. This normalization directly determines what the "untrained vs. trained" difference captures. Without an ablation (KAN with vs. without histogram normalization), it is unclear how much of the performance gain is due to the KAN's local plasticity mechanism versus the normalization itself.

### Minor

- **Near-random Age benchmark unexplained (Table 4)** — All methods, including KAN (50.5 ± 0.5), perform near chance on the Age benchmark. The paper does not discuss why. Is the InD/OOD feature overlap so large that detection is near-impossible regardless of method? Is this a fundamental limitation of OOD detection using age as the split criterion in this clinical dataset? This warrants at least a paragraph of diagnosis, as it affects the claimed generality across domains.

- **Places365 discrepancy between text and Table 1** — Section 3.1 explicitly lists Places365 as a far OOD dataset for the CIFAR benchmarks, alongside MNIST, SVHN, and Textures. However, Table 1 only shows three far OOD columns (MNIST, SVHN, Textures) — Places365 is missing. Whether this result was omitted and why should be clarified, as it affects the completeness of the reported far-OOD averages.

- **P=1 catastrophic failure and its implications** — Table 7 shows that P=1 yields 46.08 ± 15.58 AUROC with an enormous variance, far worse than random. The paper uses this to motivate partitioning, but does not adequately discuss consequences for settings where class labels are unavailable and k-means partitioning may be unreliable (e.g., regression tasks, highly imbalanced data, or datasets where the cluster structure does not align with the semantic OOD boundary). The practical robustness of k-means initialization is not analyzed.

- **No guidance on selecting P without OOD validation data** — The number of partitions P is a critical hyperparameter that varies performance by nearly 50 AUROC points. While class-label-based partitioning is intuitive for classification tasks, the paper provides no heuristic for choosing P in regression or unsupervised settings without access to OOD samples for validation.

### Tiny

- "Local neuroplasticity" is used as a central concept throughout but is never precisely defined. Section 2.1 provides an informal description; a one-sentence formal definition (e.g., quantifying the spatial extent of spline coefficient updates per sample) would sharpen the contribution narrative.

- The main text mentions Appendix A.4 for initialization stability results without summarizing the finding inline. Noting that "KAN initialization stochasticity is smaller than backbone training stochasticity" directly in Section 3.1 would help readers assess robustness without consulting the appendix.

---

## Nice-to-Haves

- **Class-conditional GMM/KDE baseline on the same backbone features** — Comparing against a GMM or kernel density estimator applied class-conditionally to the identical feature representation would isolate whether the gains are specific to spline smoothing or are achievable with any smooth density estimator. This directly addresses the conceptual question of what KANs uniquely contribute beyond class-conditional density modeling.

- **Spline coefficient difference heatmaps** — Visualizing the Δ matrix for representative InD, near-OOD, and far-OOD samples would directly validate the mechanistic claim that OOD samples activate predominantly untouched grid regions, rather than simply producing globally lower magnitudes.

- **Performance vs. computational cost trade-off as P varies** — Table 7 shows AUROC saturates around P=10; reporting wall-clock training and inference time alongside AUROC would help practitioners select P with awareness of the cost-benefit trade-off.

- **KAN used as the primary feature extractor** — Evaluating OOD detection when the backbone itself is a KAN would validate whether local plasticity benefits compound through the full pipeline, and is a natural next step given the proposed mechanism.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **CIFAR-100 non-significance (Harsh Critic):** The paper correctly handles this by bolding both KAN (83.44 ± 1.99) and NAC (83.36 ± 0.84) with the Welch's t-test criterion, explicitly acknowledging no statistically significant difference. This is not a flaw; it is accurate and transparent reporting.

- **Training task arbitrariness as a conceptual weakness (Harsh Critic):** Section 3.3 explicitly investigates this, reporting only ~0.2% difference between classification and regression-to-constant on image benchmarks. The paper is forthright about this design flexibility and provides an explanation grounded in the "coefficient registration" view of training.

- **KAN initialization stability (positive reviewer, spark finder):** Addressed in Appendix A.4. The finding — that KAN initialization stochasticity is smaller than backbone training stochasticity — constitutes a reasonable and direct response. The concern is not absent but is adequately mitigated.

- **Missing related works (Normalizing Flows, GMM, etc.) in related work section:** Per review policy, absence of related work citations is not penalized as the reviewer cannot confirm existence or relevance of external sources.

- **Demanding theoretical generalization bounds as a required contribution:** This is an empirical systems paper with a post-hoc detector. Requiring formal generalization bounds is not standard in this community setting and would be scope creep.

- **"Unfair comparison" where baseline methods use weaker configurations (Harsh Critic on various baselines):** Any comparison where KAN has a potential advantage (multi-layer features) is already flagged as a major weakness above. Separate concerns about baselines using weaker configurations than optimal are handled through that lens.

---

## Novel Insights

The most genuinely insightful observation emerging from synthesis of the reviews is the following: *the method's theoretical basis rests on an implicit assumption of sparse InD coverage of the spline grid* — that is, training data does not densely fill the entire spline input range, leaving identifiable "untouched" regions that OOD samples activate. This assumption is reasonable for low-to-moderate dimensional feature spaces where data lies on a manifold, but becomes increasingly fragile as feature dimensionality grows (high-dimensional backbone features may densely cover the grid after histogram normalization). The fact that performance peaks at a relatively small grid size G (Table 7 shows G=100 is optimal, not G=200) is consistent with this interpretation and may implicitly reflect the effective data dimensionality. This sparsity assumption is never explicitly stated, never analyzed empirically, and is the most important unresolved question about the method's scope of applicability. Papers extending this work should instrument the fraction of updated vs. untouched grid coefficients as a direct probe of whether the core assumption holds for a given dataset and backbone.

---

## Suggestions

1. **Add a computational cost table** comparing training time, inference latency, and memory usage for KAN (with varying P), KNN, NAC, and VIM. Even approximate wall-clock times on a standard GPU would be sufficient to answer this critical practical question.

2. **Explicitly clarify the feature extraction protocol for all baselines** — state whether KNN, VIM, RMDS, and other baselines are evaluated with single-layer (final) backbone features or multi-layer concatenated features, matching the KAN configuration. If baselines use single-layer features, re-run a fair comparison with uniform multi-layer features across all methods.

3. **Add a histogram normalization ablation** (KAN without normalization, or with standard z-score normalization) to disentangle the contribution of the normalization step from the KAN detection mechanism.

4. **Investigate and discuss the Age benchmark (Table 4)** — compute and report the pairwise feature-space overlap between the InD and OOD populations (e.g., via a Kolmogorov-Smirnov test on each feature) to confirm the near-chance performance is a property of the data, not the method.

5. **Resolve the Places365 discrepancy** — either add Places365 results to Table 1 or explicitly state it was excluded and why, to ensure the far-OOD averages are complete and reproducible.

6. **Measure spline coefficient update sparsity** across benchmarks as a direct empirical test of the core "untouched regions" assumption. Report the fraction of grid cells that receive significant coefficient updates during training and correlate this with detection performance to show when the method is expected to work well or fail.

---

**Evaluation summary:** The paper presents a genuinely novel and mechanistically interesting application of KAN architecture properties to OOD detection. The empirical results are largely strong — particularly the ImageNet gains and the data-size robustness — and the histogram baseline comparison provides meaningful evidence for the spline mechanism's value. However, the absence of any computational cost analysis and the unresolved multi-layer fusion fairness question are significant gaps for a deployment-oriented paper. The core claim that "local neuroplasticity drives performance" remains partially unvalidated without an ablation of histogram normalization and without quantifying spline update sparsity. On balance, the paper makes a real contribution but requires these methodological clarifications before the empirical claims can be fully trusted.