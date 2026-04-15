Now I have enough information to produce the final consolidated review. Let me synthesize everything.

---

## Summary

This paper proposes a post-hoc OOD detection method based on Kolmogorov-Arnold Networks (KANs). The core idea is to compare per-activation responses between a trained KAN and its untrained counterpart: InD samples activate trained spline regions (yielding large response differences), while OOD samples activate unmodified regions (yielding small differences). To address the base method's inability to capture joint feature distributions, the authors propose partitioning the InD dataset and training separate KAN detectors per partition, aggregating scores via the maximum function. The method is evaluated on seven benchmarks spanning image and tabular medical domains.

---

## Strengths

- **Genuinely novel detection mechanism**: Using the difference between a trained and untrained KAN as an OOD signal is a new framing compared to existing post-hoc approaches (softmax scores, feature distances, gradients). Eq. (5)'s decomposition into "where InD information is stored" (coefficient differences) and "which regions a sample activates" (B-spline masks) provides an interpretable, principled way to construct the score.

- **Strong and clear ImageNet wins**: On ImageNet-200 FS and ImageNet-1K FS, KAN achieves 71.46 and 78.52 overall average AUROC, compared to 67.18/76.25 and 76.28 for the best competitors respectively — margins of 3–4+ points that are clearly outside noise. These are the most demanding benchmarks in the suite, making these wins particularly meaningful.

- **Robustness to training data size (Table 6)**: KAN maintains ~93–94% AUROC on CIFAR-10 from 100% data all the way down to 0.1% (5 samples per class), while KNN collapses to 8.15% and VIM to 76.38% at 0.1%. This is a striking and practically important finding not shared by the leading competing methods.

- **Honest acknowledgment and demonstration of a key limitation**: Section 2.3 and Figure 3 transparently identify and illustrate the marginal-distribution limitation of the base method, and show how partitioning addresses it — this kind of clarity is valuable and reflects scientific honesty rather than papering over weaknesses.

---

## Weaknesses

### Fatal
*None.*

### Major

- **The base KAN detector (P=1) catastrophically fails, yet the paper frames the base mechanism as the primary contribution.** Table 7 shows AUROC of 46.08 ± 15.58 (below random) with P=1 on CIFAR-10. The actual working method is a *partitioned KAN ensemble*, which is meaningfully different from the base detector described in Sections 1–2.2. The paper discusses partitioning as a "limitation fix" in Section 2.3, but does not frame it clearly as the core practical contribution. More importantly, the P=1 ablation is only reported for CIFAR-10; there is no systematic comparison of partitioned vs. unpartitioned KAN across all benchmarks. It is unknown how broadly the base detector fails, making the headline mechanism story partly decoupled from what is actually driving the results.

- **Missing computational cost analysis.** The method requires training P separate KAN models (P=10 in the experiments) plus maintaining an untrained KAN, then performing P forward passes at inference. No training time, inference time, or memory overhead comparison against any baseline is provided. For a post-hoc method that competes with single-pass methods like MSP, Energy, or even KNN, this is a critical omission — especially on ImageNet-1K FS where P must scale with the number of classes.

- **Mechanism claim (KAN-specific local neuroplasticity) is not isolated from simpler alternatives.** The paper attributes performance gains specifically to KAN local plasticity. However, the only ablation comparing alternatives is the histogram baseline (85.29% vs KAN 94.12% on CIFAR-10), which shows splines beat binary bins — but this does not isolate the *KAN-specific* property from any other smoothly trainable local function approximator. An MLP trained-vs-untrained comparison, RBF features, or any non-KAN spline layer would be necessary to substantiate the "local neuroplasticity" framing rather than a simpler "any local function memorizer" explanation.

- **Overclaiming empirical dominance.** The abstract and Section 1 state "outperforms current SOTA techniques across all seven benchmarks on the overall average AUROC." This is inaccurate: Table 4 (Age benchmark) shows KAN at 50.5 ± 0.5 vs. KLM at 51.0 ± 0.7. Moreover, on CIFAR-100, KAN achieves 83.44 ± 1.99 vs. NAC's 83.36 ± 0.84 — the error bars clearly overlap, making this a statistical tie at best and not a clean win. The conclusion similarly states the method is "unaffected" by data size perturbations, while Table 6 shows a non-negligible drop from 94.12 to 93.21 (CIFAR-10) and 83.44 to 81.44 (CIFAR-100) at extreme subsampling. These are overstated; the correct characterization is *relatively more robust*, not invariant.

### Minor

- **Near-random performance on the Age medical benchmark goes unaddressed.** Table 4 shows all methods, including KAN (~50.5%), perform near random (~50%) on the Age benchmark. This represents a regime where OOD detection may be fundamentally infeasible, but the paper does not discuss why or characterize this as a known hard case. Claiming to "rank in the top three" when all top-three methods are near chance is misleading without this context.

- **Scoring function choice deferred to appendix.** The median as the aggregation function is central to the method, yet its justification is deferred to Appendix A.1, which is omitted from the submission text. The main paper offers only Figure 4 (showing a distribution for three hand-picked samples) and a brief statement about robustness to outliers. This is insufficient for such a central design choice.

- **"Seamless integration with any pre-trained classifier" is overstated.** Only two backbone families are tested: ResNet (image) and FT-Transformer (tabular). The claim of compatibility with "any" architecture or training procedure is not substantiated by this limited backbone diversity.

### Trivial

- **CIFAR-100 win over NAC is within statistical noise** (83.44 ± 1.99 vs. 83.36 ± 0.84) and should not be counted as a clean empirical win in summary statements.

---

## Nice-to-Haves

- **MLP trained-vs-untrained ablation**: Implementing the same comparison using an MLP would directly test whether gains are KAN-specific or general to the trained/untrained comparison paradigm.
- **Ensemble baseline comparison**: Comparing against P-ensemble versions of KNN or Mahalanobis would clarify whether the improvement over single-model baselines comes from KAN properties or simply from ensembling.
- **Partition count ablation across all benchmarks**: Table 7 shows P sensitivity only for CIFAR-10. Reporting P ablations for at least one ImageNet and one tabular benchmark would characterize the robustness of this critical hyperparameter.
- **Score distribution visualizations across full test sets**: Figure 4 shows three handpicked examples; full InD vs. OOD score distribution histograms would reveal whether separation is clean or driven by outlier subgroups.
- **Formal conditions or failure case analysis for near-OOD**: The method appears especially strong on far OOD; characterizing when it fails on near-OOD samples would help practitioners.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Regression results unsubstantiated" (Harsh Critic)**: The main paper explicitly states "The results, presented in Appendix A.2, demonstrate that our method also performs well in these scenarios." Per the hard rules, we do not doubt cited entities/appendices. Removed.

- **"The claim that seamless integration with any pre-trained classifier is unsupported" as a major weakness**: Kept as minor above. The limited backbone diversity is real, but the method genuinely does operate post-hoc on latent features, making the architecture-agnostic framing partially defensible. Downgraded to minor.

- **"Training-time method comparisons are missing" (Neutral Reviewer)**: The paper explicitly scopes to post-hoc methods and excludes training-time regularization (MOS, CIDER). This is intentional scope-setting, not an omission. Removed.

- **Generic strength "the paper is well-written" or "broad applicability is notable"**: Removed per hard rules on generic strengths.

- **"Only 2 backbone families tested" as a major weakness**: The test protocols are benchmark-defined — the paper uses standard OpenOOD ResNet backbones, which is the community norm. Kept only as minor.

- **"Histogram normalization under-described" (Harsh Critic)**: This is a preprocessing detail whose impact is partially addressed by the histogram baseline comparison. Too implementation-detail-focused for a major weakness.

---

## Novel Insights

The most genuinely novel observation is the training data size robustness shown in Table 6: KAN achieves ~93% AUROC on CIFAR-10 with only 5 samples per class, while KNN degenerates to 8.15% and VIM to 76.38%. This is not merely an empirical curiosity — it reflects a structural property of the method: the detector is learning a local approximation of InD support in feature space, and B-spline functions can meaningfully register a distribution from very few examples if the grid is appropriately calibrated. This property may have practical implications beyond OOD detection, potentially informing few-shot anomaly detection or class-imbalanced detection settings. The paper identifies this finding but does not fully theorize it; doing so would strengthen the contribution considerably.

---

## Suggestions

1. **Reframe the paper's contribution clearly**: State in Section 1 (and the abstract) that the *practically deployable* method is the partitioned KAN ensemble, not the base detector. Restructure Section 2 accordingly.
2. **Add a runtime/FLOPs table** comparing training and inference cost vs. NAC, KNN, and VIM across benchmark scales. This is essential for practitioners.
3. **Correct the headline claim**: Replace "outperforms across all seven benchmarks" with "achieves the best average AUROC on four of seven benchmarks (CIFAR-10, ImageNet-200 FS, ImageNet-1K FS, synthetic OOD) and ranks in the top three across the remaining three."
4. **Provide the P=1 ablation for all major benchmarks** to characterize how universally the partitioning fix is needed.
5. **Add an MLP trained-vs-untrained ablation** on at least one benchmark to test whether the mechanism is KAN-specific.
6. **Address the Age benchmark explicitly**: Note that all methods perform near-random in this setting, discuss possible causes (weak covariate shift signal after feature removal), and do not count it as a competitive result.

---

## Evaluation on Key Axes

- **Novelty**: Moderate-to-good. Using trained-vs-untrained KAN activations as an OOD signal is a genuinely original angle; the decomposition in Eq. (5) is elegant. Novelty is tempered by the mechanism not being rigorously isolated.
- **Technical soundness**: Moderate. The method is coherent and well-specified. The base detector's catastrophic failure without partitioning, and the lack of mechanism isolation, are real weaknesses in the technical story.
- **Empirical support**: Good on image benchmarks (especially ImageNet), moderate on tabular medical, with legitimate concerns about overclaiming on CIFAR-100 and Age.
- **Significance**: Moderate-to-good. The data-size robustness finding is genuinely useful. The ImageNet results suggest the method is practically competitive. The paper opens a new direction but needs refinement.
- **Clarity**: Good overall, but the framing mismatch between the base detector narrative and the partitioned ensemble reality creates confusion about what the core method actually is.

---

## Score and Decision

**Calibration against past reviews:**
- AAZ3vwyQ4X (4.0, Reject): MSPL paper with a structural evaluation flaw (F1 metric certifying degenerate results), unlabeled table rows, and no external baselines. This KAN paper is clearly better: the empirical results are genuine and clearly reported, the methodology is honestly presented, and there is no structural flaw of that magnitude.
- mMPaQzgzAN (6.5, Accept): JumpReLU SAE paper with a clean KDE-STE theoretical contribution and solid multi-site empirical validation. The KAN paper has a less rigorous mechanism story and more overclaiming, but comparable empirical breadth and a strong robustness finding.
- 8QqQk1c0Dg (6.5, Accept): Adam high-probability convergence theory paper — stronger theoretical grounding but weaker experiments.

This paper sits **between 4.0 and 6.5, closer to 5.5**. It has genuine empirical contributions (ImageNet wins, robustness to data size), an interesting novel mechanism, and honest treatment of limitations. However, the base method fails without partitioning (with this not clearly framed), the "SOTA across all 7 benchmarks" claim is inaccurate, and the missing computational analysis is a real gap. The paper needs meaningful revisions to the framing and claims, and ideally an MLP ablation and computational cost analysis, before it fully supports its stated contributions.

**Score: 5.5 — Weak Accept (borderline)**

The work is above the ICLR average in terms of empirical substance and novelty of the core idea, but the framing and overclaiming issues are consequential enough to warrant revision. The ImageNet results and data-size robustness finding provide sufficient substance for acceptance if revised accurately.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>