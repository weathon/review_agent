## Summary

This paper proposes using bottleneck adapters as task-specific feature modifiers in incremental learning, co-training them with the backbone rather than freezing it. The core idea is to let the backbone learn task-invariant features while adapters capture task-specific information, integrating this architecture into existing regularization-based methods (EWC, MAS, LwF, LwM). Empirical results on CIFAR-100 and ImageNet-Subset under Task-IL are presented, alongside ablations on bottleneck width and frozen versus co-trained backbones.

## Strengths

- **Principled architectural idea with empirical motivation.** The paper grounds its proposal in the observation that task ordering (and thus inter-task diversity) affects forgetting severity, shown in Figure 1 for LwF on CIFAR-100. This motivates the need to model inter-task differences explicitly.
- **Direct ablation validating co-training over frozen backbones.** Table 2 shows that an LwF model with adapters achieves 74.0% average accuracy when the backbone is co-trained, versus 72.9% when frozen, empirically justifying the decision to update both components.
- **Broad compatibility and method-specific integrations.** The paper derives targeted modifications for diverse existing methods: an additional backbone distillation loss $R_\varphi$ for prediction-regularized methods (Eq. 1) and exempting adapter parameters from Fisher consolidation for weight-regularized methods. It also demonstrates extensibility to modern task-specific architectures, improving DualPrompt (88.2% → 89.3%) and iTAML (79.0% → 80.1%) and outperforming TAMiL (Section 4.2, Table 2).

## Weaknesses

### Fatal

None.

### Major

- **Baseline comparisons are confounded by simultaneous algorithmic changes, so the causal attribution to adapters is unsupported.** For prediction-regularized methods (LwF, LwM), the adapter variant introduces both adapters *and* a novel backbone-distillation loss $R_\varphi$ (Eq. 1) that the baseline lacks. For weight-regularized methods (EWC, MAS), the adapter variant adds parameters *and* exempts them from regularization, effectively reducing the regularization budget. The paper never isolates these confounds—there is no test of LwF with $R_\varphi$ but without adapters, nor of weight-regularized methods with an equivalent number of unregularized non-adapter parameters. Because baseline and proposed variants differ in multiple variables at once, the central claim that adapters *specifically* drive the improvements is empirically unsupported.
- **ImageNet results contradict the paper’s claims of consistent improvement, yet the text glosses over regressions.** The abstract and conclusion state that adapters “consistently outperform non-adapter counterparts” and “consistently improve the performance of all considered methods.” However, Table 1 shows that on ImageNet-Subset, EWC-A underperforms EWC on tasks 2–5 (e.g., 76.0 vs. 80.3 after task 2, 67.7 vs. 74.6 after task 3), LwM-A underperforms LwM on every task after task 2, and LwF-A falls below LwF at task 10 (67.2 vs. 68.2). While the authors note that ImageNet hyperparameters were transferred from CIFAR-100 and training was limited to 50 epochs, they nonetheless assert that “methods with adapters yield the best performance across all incremental tasks” and “demonstrate non-trivial performance improvement.” A paper whose own data show multiple adapter variants regressing cannot credibly claim universal improvement.
- **No empirical validation of the core architectural rationale.** The entire proposal rests on the factorization that “the backbone network focuses on learning invariant features” while “adapters capture task-specific information” (Sections 3.2, 5). Yet the paper provides no analysis of feature spaces, adapter activations, or probing experiments to verify this factorization. The ablation in Section 4.3 compares frozen versus co-trained backbones, but it does not demonstrate that the co-trained backbone converges to task-invariant representations or that adapters encode task-specific ones.
- **Central claims rely almost entirely on Task-IL while broad claims are made about incremental learning generally.** The abstract and conclusion claim the approach “effectively addresses the stability-plasticity dilemma” and “eliminates the stability-plasticity dilemma for incremental learning.” However, all main experiments use Task-IL (multi-head evaluation with a task-ID oracle), a substantially easier protocol that sidesteps the cross-task discrimination challenge that Class-IL is designed to measure. Class-IL results are relegated to Appendix B and are not discussed or analyzed in the main text, which offers no evidence that the claimed benefits hold in the standard practical setting, nor any explanation of how task-specific adapters are selected at inference without task IDs.

### Minor

- **The claim that “inter-task differences are the primary driver of catastrophic forgetting” is overstated.** Figure 1 shows that task ordering affects LwF performance on CIFAR-100, with coarse-grained ordering (higher inter-task diversity) leading to more forgetting. However, the paper does not compare inter-task diversity against other factors such as capacity limits or regularization strength to establish primacy. “A primary driver” would be better supported than “the primary driver.”
- **No variance or statistical significance is reported.** The paper states results are averaged over 10 runs, but no standard deviations or confidence intervals appear in any figure or table, making it impossible to assess whether the ~1–3% gaps are reliable.
- **The bottleneck-width ablation raises a capacity alternative explanation.** Figure 6 shows that larger bottleneck widths generally yield higher accuracy. Without a capacity-matched non-adapter baseline (e.g., parallel linear layers with comparable parameter counts), it remains unclear whether adapter structure or simply added capacity drives the gains.

### Trivial

- **Notational inconsistency in Section 3.2.1.** The distillation-loss equation uses $\varphi^{t'}(x)$ and $\varphi^t(x)$ (backbone features), while the surrounding text refers to $M$ as quantifying “the similarities between the adapter outputs.” This is a minor typo-level inconsistency.

## Nice-to-Haves

- Empirical validation of the invariant/specific factorization (e.g., via Centered Kernel Alignment across task features, probing accuracy, or similarity matrices).
- Feature-space visualization (t-SNE/PCA) to show whether backbone representations become more inter-task invariant when adapters are present.
- Explicit diagnosis of the ImageNet failure modes (EWC-A, LwM-A regressions) in terms of hyperparameter sensitivity, overfitting, or architectural limitations.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Complaint about modern-method comparison protocols being unspecified.** The main text states that experiment details for the DualPrompt, iTAML, and TAMiL comparisons are in Appendix A. This is a normal space constraint; the appendix exists in the original submission.
- **Strength claim about “robust and broad accuracy gains across datasets.”** This was dropped because the ImageNet results in Table 1 show regressions for EWC-A, LwM-A, and LwF-A that contradict the claim of consistent improvement across datasets.
- **Any parser-induced formatting artifacts** (typos, line breaks, garbled text) are not author errors.

## Novel Insights

The most consequential issue in this submission is not that adapters fail, but that the experimental design makes it impossible to know whether they succeed *as adapters*. Because the prediction-regularized variant adds an extra backbone-distillation loss $R_\varphi$ and the weight-regularized variant changes the regularization budget, the paper’s positive results on CIFAR-100 could plausibly stem from these algorithmic adjustments rather than from the adapter architecture itself. This is a subtle but critical methodological gap: the paper treats an architectural modification and an algorithmic modification as a single intervention, leaving readers unable to attribute causality. Future revisions should isolate the adapter architecture via controlled ablations before claiming adapter-specific benefits.

## Suggestions

- Add ablations that isolate the adapter architecture from the accompanying algorithmic changes: (1) test LwF with the added $R_\varphi$ loss but without adapters; (2) test EWC with an equivalent number of unregularized non-adapter parameters (e.g., parallel bottleneck layers) instead of adapters.
- Move Class-IL results and the task-agnostic inference mechanism into the main body, or temper the abstract/introduction claims to reflect the Task-IL scope of the main experiments.
- Report standard deviations or confidence intervals across the 10 runs to enable statistical assessment of the reported gaps.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/5U1rlpX68A.md` (SD-LoRA, avg 7.50, Accept Oral): A strong adapter-based continual learning paper with theoretical grounding, extensive benchmarks, and well-isolated ablations. The paper under review is well below this anchor because it lacks theoretical support, has confounded comparisons, and contradicts its own claims on ImageNet.
- `/home/wg25r/review_agent/human_reviews/6r0BOIb771.md` (Bayesian MCL, avg 5.33, Reject/Withdrawn): An interesting framework with extensive empirical comparisons but limited novelty concerns. The paper under review has less novelty (adapters are well-known) and more severe experimental flaws (confounded baselines, overclaiming contradicted by data), so it should score below this anchor.
- `/home/wg25r/review_agent/human_reviews/H6pf70GZVU.md` (YoooP, avg 5.00, Reject): A prototype-based non-exemplar IL method with reasonable ideas but experimental issues. The paper under review is comparable in overall quality but its confounded comparisons are a more fundamental threat to its core claim.
- `/home/wg25r/review_agent/human_reviews/8FxELTdwJR.md` (Hyperparameters in CL, avg 4.67, Withdrawn): Criticized for confounded comparisons (incomplete DER implementation). The paper under review shares the confounded-comparison weakness and also overclaims relative to its own ImageNet results. It is arguably slightly stronger because its core idea is more concrete and CIFAR-100 results are genuinely positive, but the combination of confounding and self-contradiction places it in a similar band.
- `/home/wg25r/review_agent/human_reviews/ZyMXxpBfct.md` (Forward Explanation, avg 1.50, Reject): An extreme low anchor with unsubstantiated claims and trivial observations. The paper under review is substantially stronger than this—it presents a sensible idea and real experiments—but its central causal claims are similarly unsupported.

**Reasoning:** The paper introduces a pragmatic idea and demonstrates broad compatibility, but its central empirical claims are undermined by confounded comparisons (unable to isolate adapter effects), contradicted by its own ImageNet results, and unsupported in the standard Class-IL setting. These are serious methodological and evidentiary gaps that place it below the medium-scoring Bayesian MCL and YoooP anchors, and in the vicinity of the withdrawn Hyperparameters in CL paper. It is not as weak as the 1.5–3.0 band because the CIFAR-100 results are real and the frozen-vs-co-trained ablation is useful. A score of 4.5 reflects a paper with genuine merit that requires major experimental revision to support its claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>