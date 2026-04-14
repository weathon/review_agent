## Summary
This paper proposes a novel OOD detection method based on Kolmogorov-Arnold Networks (KANs), leveraging the local support property of B-spline activation functions ("local neuroplasticity"). The core idea compares the activation patterns of a trained KAN against an identically initialized untrained KAN: InD samples activate spline regions that were adapted during training (producing a large difference), while OOD samples activate unadapted regions (producing a small difference). To handle the inherent limitation of capturing only marginal feature distributions, the authors introduce a dataset partitioning strategy with multiple KANs trained on subsets of the data. The method is evaluated on seven benchmarks spanning image classification (CIFAR-10/100, ImageNet-200/1K full-spectrum) and medical tabular data (eICU Ethnicity, Age, and Synthetic OOD), reporting state-of-the-art overall average AUROC and notable robustness to reductions in training dataset size.

---

## Strengths

- **Principled and novel mechanism with mechanistic interpretability:** The decomposition in Eq. 5 — where `|c_trained − c_untrained| · B_i(x_p)` separates "where InD information is stored" from "which region the test sample activates" — provides an interpretable, first-principles motivation absent from most distance-based or gradient-based OOD detectors. The toy visualizations (Figures 2–3) clearly illustrate the mechanism.

- **Strong ImageNet performance:** On ImageNet-200 FS, the KAN detector achieves 71.46 ± 0.40 overall average AUROC vs. NAC's 60.05 ± 0.58 — an approximately 11-point improvement over the previously strongest post-hoc method. This gap persists on ImageNet-1K (78.52 vs. 76.28). These are the most compelling empirical results in the paper.

- **Exceptional data-efficiency:** Table 6 shows the KAN detector maintains 93.21 ± 0.53 AUROC on CIFAR-10 at 0.1% training data, while KNN collapses to 8.15% and VIM degrades to 76.38%. This robustness to training-set size is a concrete and practically significant property, not shared by the strongest baselines.

- **Histogram ablation substantiates KAN's advantage over simple counting:** Replacing splines with binary histograms achieves 85.29% vs. 94.12% for the KAN, providing direct evidence that the smoothness of B-splines (not just the trained-vs-untrained comparison strategy) contributes meaningfully to performance.

- **Medical tabular improvements:** On Synthetic OOD, the KAN detector achieves 79.13 ± 2.22 average AUROC vs. 75.63 ± 1.26 for the best baseline (MDS), and 61.4 ± 3.1 on the Ethnicity benchmark vs. 58.5 ± 0.2 for MDS — both meaningful improvements in a challenging domain.

---

## Weaknesses

- **Critical missing ablation — MLP trained-vs-untrained comparison:** The paper attributes performance gains specifically to "local neuroplasticity" of KAN's B-splines, but no experiment compares the proposed detector using a standard MLP or any other architecture with locally-supported activation functions (e.g., RBF networks). The histogram ablation shows KAN > simple counting, but the question of whether the improvement comes from the *architecture* (B-spline locality) or from the *detection strategy* (comparing trained vs. untrained activations) remains unanswered. Without this control, the core claim — that KAN-specific local neuroplasticity is the source of the advantage — is unsubstantiated. This is the most important missing experiment.

- **Partitioning trick is not optional — it is essential:** Table 7 shows P=1 achieves 46.08 ± 15.58 AUROC (essentially random with high variance), while P=10 achieves 94.12. This means the base KAN detector without partitioning is non-functional; the partitioning is a required component, not a refinement. The paper presents it as an improvement to "overcome a limitation," but the performance gap indicates it is absolutely necessary. This deserves a more prominent and explicit discussion.

- **Marginal-only density estimation is a fundamental limitation with only a heuristic fix:** As the paper itself acknowledges in Section 2.3, each KAN activation function receives only one input, so the detector captures marginal rather than joint feature distributions. The partitioning strategy approximates joint density via class-conditional marginal densities — conceptually similar to class-conditional Mahalanobis distance. There is no theoretical guarantee on how many partitions suffice, and the method provides no convergence guarantee to joint density estimation as P grows.

- **No computational cost analysis:** The method requires training P separate KANs at setup time (P=10 for CIFAR-10, potentially P=1000 for ImageNet-1K if using class-label partitioning) and running two forward passes at inference. KANs are generally slower than MLPs due to spline evaluations. Given the paper explicitly claims suitability for "real-world scenarios" and "automotive contexts," the absence of inference latency and training time comparisons against NAC, KNN, and VIM is a significant omission that undermines the practical deployment claims.

- **Histogram normalization lacks ablation:** Section 3.1 notes that a histogram normalization step is applied to address skewed latent features, so that samples span the KAN's grid range. This preprocessing step is not present in most baselines and could itself account for part of the performance gains. An ablation removing this normalization step is needed to isolate its contribution.

- **Multi-layer feature integration ablation is missing:** The paper directly adopts multi-layer backbone feature concatenation from NAC (Liu et al., 2024a). Whether the KAN detector's advantage over NAC persists without this trick — i.e., using only the final backbone layer — is not tested. This is important because the paper needs to show that the KAN architecture contributes beyond the feature extraction strategy it borrows from its strongest competitor.

- **Overstatement of results in abstract and conclusion:** The abstract claims the method "demonstrates superior performance and robustness compared to state-of-the-art techniques" across all seven benchmarks. On CIFAR-100 (Table 1), KAN achieves 83.44 ± 1.99 vs. NAC's 83.36 ± 0.84 — statistically indistinguishable under the paper's own Welch's t-test criterion, as both are bolded. On the Age benchmark (Table 4), KAN achieves 50.5 ± 0.5 while KLM achieves 51.0 ± 0.7 — also overlapping confidence intervals. The abstract should be qualified to "outperforms on overall average AUROC" rather than claiming universal dominance.

---

## Nice-to-Haves

- Analyze why the KAN advantage over NAC is ~11pp on ImageNet-200 but only ~0.08pp on CIFAR-100. Is this driven by scale, number of partitions, or feature dimensionality? Such an analysis would clarify when the method is most useful.
- Extend the partition hyperparameter ablation (Table 7) to ImageNet and medical benchmarks, not just CIFAR-10. This would establish transferability of the P=10 choice.
- Visualize the Δ matrix heatmaps for representative InD and OOD samples to verify that differences are concentrated in specific spline regions as claimed by the local plasticity hypothesis.
- Report per-OOD-dataset AUROC (not just averages) for at least one main benchmark, to detect potential failure modes hidden by averaging.
- Discuss grid boundary behavior: for OOD test samples that fall outside the spline grid's training range, B-spline extrapolation may activate edge coefficients for both InD and OOD samples indiscriminately. Even a brief discussion of this edge case would strengthen the theoretical characterization.
- Include an AUROC vs. inference time plot to quantify the efficiency-accuracy trade-off relative to simpler baselines.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **Age benchmark as a specific failure of the KAN method:** The harsh critic characterizes the Age benchmark result (50.5 ± 0.5 AUROC) as "a significant failure mode" of the KAN detector. However, examining Table 4, every single baseline also performs at essentially chance level (MDS: 50.8, RMDS: 48.3, KNN: 49.6, VIM: 48.8, SHI: 50.4, KLM: 51.0, OpenMax: 48.1). This is a benchmark-wide phenomenon — the age feature is removed per the benchmark protocol, making the OOD split essentially invisible to all post-hoc feature-space detectors. Singling this out as a specific KAN weakness is unfair.

- **KNN's 8.15% at 0.1% training data is a "typo":** The harsh critic suggests the 8.15 ± 0.86 value for KNN at 0.1% CIFAR-10 is a typo. In reality, at 0.1% of CIFAR-10 training data, only ~5 samples per class are available. KNN-based detection requires a sufficient density of neighbors; with so few samples, the method can completely invert its scoring and fail catastrophically. The value appears genuine, not a typo.

- **"Contributions not listed explicitly" as a weakness:** Pure formatting/style nitpick. The contributions are clear from reading the paper, and the lack of a bulleted list is not a substantive issue.

- **Figure 1 caption as internally inconsistent:** While the figure does show an InD sample going into the Untrained KAN and an OOD sample into the Trained KAN (which is a slightly confusing choice of illustration), the corrected caption at line 39 clarifies the actual mechanism. This is at most a minor presentation issue, not a substantive flaw.

- **Criticism that the method should compare to normalizing flows or FAISS-KNN with class conditioning:** Invoking specific external references as required comparisons constitutes potential fabrication of missing related work.

- **"Local neuroplasticity" as a misleading/inflated concept:** The paper defines the term precisely and consistently in the context of B-spline local support. Whether the neuroscience borrowing is optimal branding is a stylistic opinion.

---

## Novel Insights

The most genuinely novel observation, spanning all three reviews, is the following: the KAN detector's partitioning strategy transforms the problem from marginal density estimation into an ensemble of class-conditional marginal density estimators — and the performance collapses (P=1, AUROC ≈ 46%) or succeeds (P=10, AUROC ≈ 94%) based entirely on whether this transformation is applied. This suggests the paper has actually discovered something interesting: that local-support activation functions combined with class-conditional partitioning can approximate joint density estimation with empirical performance competitive with, and on ImageNet substantially better than, explicit density-based methods. The B-spline smoothness (vs. histogram baseline: ~9% gain) ensures that the "density estimate" interpolates gracefully between training points, which likely explains the remarkable data-efficiency (performance preserved down to 0.1% training data). Whether this insight is uniquely tied to KANs or would generalize to other locally-supported architectures remains the key open question.

---

## Suggestions

1. **Add an MLP baseline** using the identical "trained vs. untrained activation difference" strategy (e.g., measuring neuron activation magnitudes before and after training). This single experiment would validate or refute the paper's core architectural claim and is essential for the submission to be accepted at a top venue.

2. **Add an inference-time benchmark table** reporting milliseconds per sample and GPU memory for the KAN detector, NAC, KNN, and VIM. If overhead is significant, discuss whether distillation of the untrained KAN response (e.g., a closed-form approximation) can reduce it.

3. **Revise abstract and conclusion** to say "outperforms on overall average AUROC" rather than "across all seven benchmarks," and acknowledge that CIFAR-100 results are statistically tied with NAC.

4. **Add a histogram normalization ablation** (with vs. without normalization for both KAN and at least one baseline) to clarify whether this preprocessing step independently contributes to the performance advantage.

5. **Provide an ablation of multi-layer feature integration** (single last layer vs. multi-layer concatenation) to determine how much of the KAN detector's advantage over NAC comes from the KAN architecture vs. the shared feature construction strategy.

6. **Add a dedicated discussion of failure modes and limitations**, particularly the sensitivity to P=1 (near-random behavior) and the reliance on the backbone to extract OOD-relevant features (if the backbone has "laundered out" the OOD signal, the KAN detector cannot recover it, as evidenced by the Age benchmark).

---

**Novelty:** Moderate-to-high. Applying KANs to OOD detection via the trained-vs-untrained comparison is a genuinely novel framing not seen in prior work. However, the novelty is partly contingent on the undemonstrated claim that the KAN architecture specifically (rather than the comparison strategy) drives performance.

**Technical soundness:** Moderate. The mechanistic derivation (Eq. 5) is clean and correct. The partitioning idea is sensible but theoretically ungrounded. The grid boundary behavior and the primacy of partitioning (P=1 = random) represent gaps in the technical story.

**Empirical support:** Moderate-to-strong. Seven benchmarks is comprehensive. The ImageNet gains are compelling. The data-efficiency result is distinctive. The CIFAR-100 gains are negligible. The missing MLP ablation leaves the central empirical claim uncertain.

**Significance:** Moderate. The data-efficiency result and ImageNet performance are practically important. The method's complexity (training P KANs, dual forward passes) relative to lightweight baselines like energy score or KNN is not quantified.

**Clarity:** Good. The paper reads cleanly, figures are illustrative, and the mathematical exposition is accessible. Minor issue with Figure 1's depiction of the comparison mechanism.

MY FINAL SCORE: <pineapple>5.8</pineapple>