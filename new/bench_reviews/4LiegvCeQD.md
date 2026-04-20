After thoroughly reading the paper and verifying reviewer claims, let me now write the consolidated review.

## Summary

The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method for single-sample inference that dynamically selects the most confident model (for the majority-voted class) as a soft target and minimizes cross-entropy across all ensemble members. Evaluated across CIFAR-10C, CIFAR-100C, and ImageNet-C with frozen batch normalization and batch size 1, IEL demonstrates consistent majority-vote and individual-model accuracy improvements on most corruption types while simultaneously improving individual classifiers—validating the "intra-model learning" claim.

## Strengths

- **Single-sample adaptation without batch statistics**: Unlike TENT and similar TTA methods that rely on batch-level statistics, IEL operates with batch size 1 and frozen BN parameters, making it applicable to deployment settings where batch statistics cannot be reliably estimated. The experimental design (Section 4, frozen BN) properly isolates the IEL contribution.

- **Broad and consistent empirical improvements across three datasets**: Tables 1–3 show IEL improving majority-vote accuracy on 12/15 CIFAR-10C corruptions, 11/15 CIFAR-100C corruptions, and all 15/15 ImageNet-C corruption types, using 4–5 diverse architectures per dataset. The ImageNet-C results with zero catastrophic forgetting across all corruption types are particularly strong.

- **Dynamic, bidirectional knowledge transfer via majority-voted soft target**: The formulation (Equation 1) implements a mechanism where the "teacher" model changes sample-to-sample based on which model has highest confidence for the majority-voted class, differing from both standard Knowledge Distillation (fixed teacher) and static ensemble voting. The observation that the self-term δ(hθ_j, hθ_j) reduces to the selected model's own entropy is a clean mathematical property.

- **Entropy minimization as a side effect rather than explicit objective**: Figure 1 demonstrates a positive correlation between IEL loss and Shannon Entropy across 15 ImageNet-C corruption types, showing that minimizing inter-model cross-entropy naturally reduces prediction uncertainty without directly optimizing entropy—addressing concerns raised in prior TTA work (Lee et al., 2024).

## Weaknesses

### Fatal
None.

### Major

- **No comparison to established TTA methods configured for the single-sample setting**: The paper positions IEL within the TTA literature (Sections 1, 2.2) and claims it is a "natural choice for Test Time Adaptation," yet compares exclusively against a static (no-adaptation) ensemble baseline. Methods like EATA (which uses augmentation-averaged predictions), COTTA (stochastic weight restoration), and ROID (ensemble of model snapshots) are mentioned in the related work but not included as baselines. While the paper argues that TENT "become[s] ineffective with access to only a single sample per batch" (Section 1), this claim is asserted without empirical demonstration. Without comparing against single-sample-configured TTA methods, it is impossible to determine whether IEL outperforms, matches, or underperforms existing approaches—leaving the paper's contribution relative to the state of the field unquantified.

- **Best-epoch cherry-picking without an online stopping criterion**: Tables 1–3 report the "highest accuracy improvements (%) over all epochs" of adaptation. In a true streaming deployment where one sample arrives at a time, validation-based epoch selection is impossible. The paper runs 5–25 epochs (visible in Figure 3) and reports peak performance, but offers no deployment-relevant metric such as last-epoch accuracy, online cumulative performance, or automatic termination heuristic. Without a stopping criterion, the reported gains may not be achievable in practice when the optimal epoch is unknowable at deployment time.

- **Tables report only relative improvements, masking absolute performance floor**: All three results tables report percentage deltas over the static baseline, not absolute accuracies. A "+20%" improvement over a model at 10% accuracy is fundamentally different from the same improvement over a model at 60% accuracy. Without absolute baselines, readers cannot judge real-world utility, and percentage improvements on low-performing models can appear misleadingly large. The paper's claim that IEL "reduces generalization error significantly better" cannot be properly assessed without knowing the absolute accuracy levels.

### Minor

- **Confirmation bias and cascading errors are acknowledged but unmitigated**: The paper acknowledges that "if the ensemble prediction is incorrect, then allowing it to supervise member models could lead to erroneously optimizing strong models on incorrect predictions" and names this as catastrophic forgetting (Section 1, Section 3). However, the method does not employ any mitigation strategies (weight resetting, memory replay, adaptive learning rates, or statistical distance measures that output 0 for identical distributions). The severe degradation on noise corruptions in Tables 1–2 (e.g., –31.01% for VGG11 on Gaussian Noise in CIFAR-10C; –15.67% on Impulse Noise in CIFAR-100C) confirms this vulnerability. The authors propose these as "future work," but the mechanism's failure modes are a core risk rather than a peripheral concern.

- **Learning rate and regularization constant lack principled justification**: The paper uses a learning rate of 0.001 and a regularization constant α = 10e⁻¹¹, which is described as "effectively makes our learning rate even smaller" (Section 4). While this small learning rate is defensible for test-time adaptation to avoid catastrophic updates, the choice of α is not justified by any analysis or ablation, making it difficult for practitioners to replicate or adapt the method.

### Trivial

- None beyond standard presentation improvements that should be addressed but carry no evaluative weight.

## Nice-to-Haves

- An automatic online stopping criterion (e.g., entropy plateau detection, consecutive-epoch disagreement tracking) would strengthen the paper's deployment claims. The paper already notes "once accuracies on the testing set start to diminish, one could terminate IEL" (Section 3.1), making this a natural extension.
- Including absolute accuracies alongside percentage improvements would improve interpretability without requiring additional experiments—the data from the static baseline runs already exists.
- Error analysis on the noise corruptions where IEL degrades performance (Gaussian Noise, Shot Noise, Impulse Noise in CIFAR) would help the community understand the method's failure boundaries.

## Removed Points

The following points were flagged but removed with justification:

- *Criticism: "The experimental protocol directly contradicts the paper's core premise of single-sample streaming."* — **Weakened, not removed entirely.** The paper does process one sample at a time per epoch (Algorithm 1); running multiple epochs is an experimental design choice, not a fundamental contradiction. The concern was recast as the more specific "cherry-pick best epoch without stopping criterion" point under Major weaknesses.

- *Criticism: "The dynamic teacher selection mechanism inherently amplifies confirmation bias."* — **Weakened.** The paper explicitly acknowledges this risk in Section 1 ("models are known to be confidently incorrect at times") and Section 3 ("we risk overwriting the strong knowledge of the majority voted models"). The failure is real, but the criticism that the paper "does not mitigate it" is partially softened by the honest reporting of degradation in Tables 1–2 and explicit future-work discussion. This appears as a Minor weakness rather than Fatal/Major.

- *Criticism: "Cherry-picking inflates gains and makes results non-reproducible."* — **Factually partially incorrect.** Best-epoch reporting is reproducible if the training setup is fixed; the real issue is deployment relevance, not reproducibility. Reframed as the "no online stopping criterion" point.

- *General criticisms about missing appendix, missing proofs, or absent references.* — **Removed per hard rules.** The parser strips appendix sections.

- *Criticism questioning whether cited models/baselines exist or are released.* — **Removed per hard rules.** All cited methods and datasets are treated as real.

## Novel Insights

The paper's central contribution—a dynamic, bidirectional teacher-selection mechanism where ensemble members learn from each other rather than from a fixed supervisor—is a genuinely different perspective on test-time adaptation. Unlike entropy minimization (which can cause confident errors) and static ensemble voting (which doesn't improve individual models), IEL creates a self-reinforcing signal that simultaneously reduces diversity and improves accuracy. The finding that this mechanism works reliably on structural corruptions (blur, pixelate, elastic transform, zoom) but degrades on noise-type corruptions (Gaussian, shot, impulse noise) is a valuable observation: it suggests the ensemble consensus captures shift-aligned *structural* features well but is vulnerable to per-pixel noise that corrupts confidence estimates. This failure pattern is informative for the broader TTA community.

## Suggestions

- Include at least 2–3 established TTA methods (e.g., EATA, NOTE, SAR) reconfigured for batch size 1 with frozen BN, to establish IEL's relative standing in the TTA landscape. Even preliminary results would significantly strengthen the paper.
- Report absolute accuracies alongside percentage improvements in Tables 1–3.
- Add a last-epoch accuracy column or an online stopping criterion analysis (e.g., termination at entropy plateau) to support the single-sample streaming deployment claim.
- Consider replacing α = 10e⁻¹¹ with a principled schedule or providing an ablation over its values.

---

**Calibration and Scoring Rationale:**

I compared this paper against several anchors:
- **High-scoring anchor**: DeYO (9w3iw8wDuE, scores 8,6,6,8, Accept Spotlight) had similarly broad experimental results across datasets and a novel framing. This paper's breadth is comparable, but DeYO had stronger baseline comparisons.
- **Mid-range anchor**: Chq4OQ3p18 (scores 5,3,8,5, Reject) proposed a simple TTA improvement with honest reporting of limitations—similar pattern but weaker experiments.
- **Reject anchor**: HyRe (8HQS1X2AK4, scores 5,5,6, Reject) had a simple ensemble formulation with missing baselines—the most structurally similar paper. Its missing-baseline issue parallels this paper's but its experiments were less comprehensive.
- **Low-scoring anchor**: Preference learning paper (fTdhM7q1o2, scores 3,3,3,3) was rejected for missing baselines and limited experiments—this paper is significantly stronger in experimental coverage.

This paper sits between HyRe (missing baselines, simple formulation) and DeYO (broad experiments, novel framing). Its experimental breadth is genuinely stronger than many borderline papers, and the honest reporting of failure modes is commendable. However, the total absence of competitive TTA baselines and the best-epoch cherry-picking are substantive flaws. The paper is better than the 3-scored papers (real experiments, real results across 3 datasets) but clearly below the 7–8 papers (proper baselines, complete evaluation protocols).

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>