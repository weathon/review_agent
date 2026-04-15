Now let me read the referenced human reviews for calibration:Now I have sufficient information to write the consolidated review. Let me synthesize.## Summary

This paper proposes a post-hoc OOD detection method built on Kolmogorov-Arnold Networks (KANs). The core idea is to compare activation patterns between a trained KAN and an identically initialized but untrained copy: in-distribution samples activate regions whose spline coefficients were modified during training and thus diverge significantly from the untrained model, while OOD samples activate untouched regions and produce similar responses in both networks. To handle the KAN's inability to model joint feature distributions (since each spline operates on a single input), the authors introduce a dataset partitioning/ensemble scheme that trains separate KANs per cluster. The method is evaluated on seven benchmarks spanning image classification (OpenOOD CIFAR-10, CIFAR-100, ImageNet-200 FS, ImageNet-1K FS) and tabular medical data (Ethnicity, Age, Synthetic OOD from the eICU dataset).

---

## Strengths

- **Novel and well-motivated detection mechanism**: Using trained-vs-untrained local basis responses as an OOD signal is a genuine and creative departure from score-based (softmax), distance-based (KNN, MDS), or gradient-based (NAC) approaches. The formalization via Eq. (5)—where the coefficient delta acts as a trained-region mask and the B-spline basis acts as a test-sample mask—is clean and interpretable.

- **Robust performance under reduced training data**: Table 6 demonstrates a qualitatively different behavior from all three strong baselines. KAN maintains 93.21–94.12% AUROC on CIFAR-10 across the full range from 100% to 0.1% training data, while KNN catastrophically degrades to 8.15% at 0.1% and VIM drops to 76.38%. This robustness property is not merely a quantitative improvement—it is a qualitatively different failure mode, and it is a concrete advantage in data-scarce settings.

- **Clear empirical leadership on large-scale image benchmarks**: On ImageNet-200 FS and ImageNet-1K FS (Table 2), the KAN detector leads across all five individual OOD subsets and all three aggregate metrics, with margins over the next best method ranging from 2–5 AUROC points on far-OOD and 1–2 points on near-OOD. This is a convincing result on the most challenging benchmarks.

- **Honest identification and explicit treatment of a real limitation**: Section 2.3 explicitly acknowledges that the KAN detector can only capture marginal distributions, not the joint distribution, and demonstrates the failure mode with both a toy L-shaped dataset and an actual ablation (Table 7). The paper does not hide the limitation.

- **The ablation study (Table 7) is genuinely informative**: Showing the dependence on partition count in a table visible to readers, rather than selecting the best P and presenting only the final number, is a form of transparency that many papers avoid.

---

## Weaknesses

### Fatal
*(none that unconditionally sink the paper)*

### Major

- **The base detector fails badly without partitioning, fundamentally complicating the central narrative.** Table 7 shows P=1 achieves 46.08 ± 15.58 AUROC on CIFAR-10—essentially chance performance with extremely high variance. Only with P=10 does the method reach the reported 94.12%. This means the method that achieves state-of-the-art results is not the "trained-vs-untrained KAN" detector described and motivated in Sections 1 and 2.2—it is a partitioned ensemble of 10 such detectors. The paper presents local neuroplasticity as the operative mechanism, but the bare mechanism fails to function on CIFAR-10 without ensemble partitioning. This is not a quibble about an optional enhancement: the paper's own data shows P=1 is broken. The authors should acknowledge this in the main text and reframe the contribution as primarily an ensemble scheme exploiting KAN marginally-local properties, rather than presenting the ensemble as a supplementary extension to handle complex cases.

- **The mechanism claim is not isolated from alternative explanations.** The paper attributes its performance to KAN's unique local neuroplasticity. However, the operational detector uses: (i) deep backbone features (not raw inputs), (ii) histogram normalization, (iii) multi-layer feature integration, and (iv) ensemble of 10 KANs trained on k-means clusters. Under this setup, the detector could be functioning primarily as an ensemble of marginal support estimators in backbone feature space, with the KAN architecture providing smooth interpolation between training points (rather than a crisp binary boundary). No ablation compares against a non-KAN ensemble of per-feature support estimators (e.g., kernel density estimation per feature, per partition) with identical scoring and aggregation. Without this, the causal attribution to "local neuroplasticity specifically" rather than "smooth per-feature support estimation in backbone features with ensembling" cannot be assessed.

- **The "state-of-the-art across seven benchmarks" claim is overstated.** Verified against the tables: (a) On CIFAR-100, KAN achieves 83.44 ± 1.99 vs NAC's 83.36 ± 0.84—the margin is within noise; (b) On the medical Age benchmark (Table 4), KAN (50.5 ± 0.5) is not the best method—KLM (51.0 ± 0.7) and MDS (50.8 ± 1.1) both exceed it, and all methods including KAN are near random chance; (c) The paper's use of Welch's t-test to declare statistical significance over only three seeds is meaningful for image benchmarks but adds false precision to the broader claim. The accurate statement is that KAN has the best overall average AUROC on the image benchmarks and is competitive but not dominant on medical benchmarks.

### Minor

- **Robustness to training set size claim is overgeneralized.** Table 6 covers only CIFAR-10 and CIFAR-100. The conclusion in Sec. 1 and Sec. 3.2 extends this to "across all considered benchmarks," but no training-size experiments are conducted on ImageNet-200 FS, ImageNet-1K FS, or any tabular benchmark. The claim should be scoped to CIFAR-10/100.

- **Multi-layer feature integration is not ablated independently.** The paper borrows multi-layer backbone feature integration from NAC and uses it as a default setting. Since NAC (the strongest baseline) also uses this technique, the comparison may be appropriate; but no ablation in the main text confirms whether KAN with single-layer features still outperforms NAC with single-layer features, or whether the multi-layer feature set is doing significant lifting independent of the KAN scoring scheme.

- **Computational overhead not quantified.** Training P=10 separate KAN detectors, storing them, and performing P+1 forward passes per sample at inference is a non-trivial overhead compared to truly post-hoc methods that require no additional training (Energy, MSP, ODIN). No training time, inference latency, or memory footprint analysis is provided. This is not fatal but is relevant for practitioners considering adoption.

### Trivial

- The claim that KANs "effectively address the curse of dimensionality" (Sec. 2.1) is an overclaim about the Kolmogorov-Arnold representation theorem that does not hold in finite-sample practice. This is a standard framing in the KAN literature but should not be taken as a validated property in this deployment context.

- The phrase "seamlessly integrated with any pre-trained classifier, regardless of model architecture, training procedures, or types of OOD data" (Sec. 3.1) is too strong given that evaluation covers only ResNet and FT-Transformer backbones. Rewording to "applied post-hoc to pre-trained backbones that expose latent features" would be more accurate.

---

## Nice-to-Haves

- **Controlled comparison isolating the KAN contribution**: Replace KAN with matched-capacity per-feature kernel density estimators or per-feature B-splines without the KAN architecture, under identical partitioning, scoring, and aggregation. This would directly test whether the architecture matters or whether any smooth per-feature estimator suffices.

- **Analysis of partition count selection**: Table 7 shows a massive jump from P=1 to P=5 (46→90 AUROC) but provides no guidance on how P should be selected for a new dataset. A discussion relating P to feature dimensionality or class count would be practically useful.

- **t-SNE or similar visualization** showing how partitioned KAN support regions look in feature space for a real benchmark (vs. only the L-shaped toy), to build intuition for why partitioning helps in practice.

- **FPR@95 in main tables**: AUROC is the reported primary metric, but FPR@95 is standard in OOD detection for safety-critical applications (particularly relevant given the medical domain). The paper reports it in Appendix A.6 but not the main text.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they may reflect reviewer misreading or out-of-scope demands.*

- **"Multi-layer features give KAN an unfair advantage over baselines" (Spark reviewer)**: Removed because the paper explicitly cites NAC for this design choice, and NAC—the strongest baseline—also uses multi-layer features. The playing field appears level on this dimension.

- **Vision Transformer evaluation (Human Finder reviewer)**: Removed as a weakness; this is an out-of-scope demand for a paper evaluated on the OpenOOD benchmark protocol, which specifies ResNet backbones. Retained as a nice-to-have at most.

- **"Narrow scope of OOD types—adversarial/corruptions not evaluated" (Human Finder reviewer)**: Removed. The paper explicitly evaluates the full-spectrum benchmarks that include covariate-shifted InD samples, and semantic-shift OOD detection is the paper's stated scope. Evaluating adversarial perturbations is out of scope.

- **"Weak theoretical justification" as a formal weakness (Human Finder reviewer)**: Weakened to minor/nice-to-have because the empirical paper genre in the OOD detection community does not require theoretical proofs; toy examples and ablations are the field norm. The reviewer's concern is real but is not a grounds for rejection.

- **Histogram normalization not analyzed in detail (Harsh Critic)**: Removed as a reprodicibility nitpick; the preprocessing step is standard and its presence is disclosed.

---

## Novel Insights

The most distinctive empirical observation in this paper—underappreciated even in the reviewers' comments—is that the robustness to training dataset size (Table 6) appears to be a structurally different phenomenon from ordinary performance stability. Methods like KNN and VIM store or reference training samples directly, so degradation with fewer samples is unsurprising. NAC, which uses gradient-based neuron activation scores, also degrades because it depends on the network being well-calibrated over many samples. The KAN detector, by contrast, only needs to register samples into spline coefficient regions—a process that appears much less dependent on sample density once P partitions are sufficient to cover the relevant cluster structure. This suggests that spline-coefficient registration may be a fundamentally more sample-efficient form of support estimation than distance or activation statistics. Whether this is due specifically to spline smoothing or more generally to the ensemble-of-marginals architecture is the open question the paper does not resolve.

---

## Suggestions

1. **Reframe the base method vs. the operational method explicitly**: State clearly in the abstract and introduction that the deployed method is a partitioned ensemble of KAN detectors, not a single trained-vs-untrained KAN, and motivate the partitioning as an essential (not optional) component of the contribution.

2. **Add a non-KAN ensemble baseline**: Train P independent per-feature kernel density estimators or spline regressors (not organized as a KAN) with identical partitioning/aggregation to isolate whether the KAN architecture specifically drives performance or whether the ensemble of marginal estimators pattern is sufficient.

3. **Narrow the SOTA claim to match the data**: Change "outperforms across all seven benchmarks on the overall average AUROC" to "achieves the best overall average AUROC on image benchmarks and is competitive on medical tabular benchmarks, with the exception of the Age benchmark where all methods are near chance."

4. **Move training-size robustness claim to be benchmark-specific**: Limit the robustness claim to CIFAR-10/100 and add a brief note that it has not yet been verified on ImageNet-scale or tabular datasets.

5. **Report inference time for P=10 vs. simpler post-hoc methods** (MSP, KNN, Energy) to give practitioners a cost-benefit picture.

---

## Score and Decision

**Calibration against anchor papers:**

- **NAC (SNGXbZtK6Q, scores 5/8/6/8, accepted spotlight)**: NAC had a cleaner mechanism story (neuron activation coverage), more extensive evaluation without structural failures in ablation, and the comparison baseline throughout this paper. KAN beats NAC on most image benchmarks but has the P=1 structural failure and overstated mechanism claims. KAN is substantively below NAC in rigor.

- **SCALE (RDSTjtnqCg, scores 5/8/6/6, accepted poster)**: SCALE is an incremental but tightly argued post-hoc method with a clean mechanism analysis. Similar difficulty level to this paper; KAN is more novel in concept but less rigorous in mechanism isolation.

- **NAP (YMgMGPjUPg, scores 5/8/3/3, withdrawn)**: NAP had novel intuition but failed on rigor and narrow evaluation. This paper has broader and stronger empirical coverage than NAP, but shares the mechanism-isolation problem.

- **Feature Map Matters (ZrY38sUYWs, scores 5/6/6/5, rejected)**: Comparable level of empirical contribution with similar mechanism questions. Those papers were scored around 5.5 and rejected.

- **HAct (Oo5spZRpH6, scores 3/5/3, rejected)**: This paper is clearly above HAct due to broader evaluation and honest ablations.

**Positioning**: The paper sits between SCALE (accepted poster, ~6.25) and NAP (rejected, ~4.75). The strongest single failure—P=1 yielding 46% AUROC on CIFAR-10—is a genuine structural problem that undermines the framing, but the results on image benchmarks are real and the robustness finding is genuinely interesting. The writing overclaims pull it below poster acceptance threshold. I position this at **5.0**—marginally below acceptance, requiring the structural reframing and additional ablation before it would be publishable in this form.

- **Novelty**: Medium-high. KAN-based OOD detection is new; the specific local-plasticity framing is creative.
- **Technical soundness**: Medium. The bare mechanism fails without partitioning; mechanism isolation is absent.
- **Empirical support**: Medium-high on image benchmarks; medium overall given the Age benchmark near-chance results and CIFAR-100 tie.
- **Significance**: Medium. Useful empirical contribution; the robustness property is genuinely valuable if confirmed on more benchmarks.
- **Clarity**: Medium. The base method / operational method distinction is blurred, and overclaims in the framing detract from otherwise adequate writing.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>