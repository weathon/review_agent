Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

The paper proposes ARC (Adaptive Retention & Correction), a test-time approach to mitigating classifier bias in continual learning. ARC leverages the overlooked availability of past-task data during inference—first identifying past-task samples via an Out-of-Task Detection (OTD) method based on confidence and predicted class, then applying two complementary corrections: (1) Adaptive Retention, which re-balances the classifier using pseudo-labeled past-task samples via cross-entropy + entropy minimization, and (2) Adaptive Correction, which revises potentially misclassified predictions using a Task-based Softmax Score (TSS). ARC is plug-and-play, requiring no training modifications, and is shown to improve 8 existing CL methods across 3 benchmarks with average accuracy gains of 2.7–3.5%.

## Strengths

- **Novel test-time perspective on classifier bias**: The core insight—that past-task data is naturally available during CL evaluation but has been systematically overlooked—is genuinely valuable. Unlike prior de-biasing methods (BiC, IL2M, OBC) that require stored memory during training, ARC exploits this test-time access to correct bias without any memory buffer (Sections 1, 3.4).

- **Broad and consistent empirical improvements**: Table 1 demonstrates consistent accuracy gains and forgetting reductions across 8 diverse CL methods (4 memory-based, 4 memory-free) and 4 dataset/split combinations. Average gains range from 2.7% to 3.5% in accuracy and 4.8% to 8.0% in forgetting reduction, with the strongest improvements on long-sequence settings (Section 4.2).

- **Clear problem decomposition via OTD**: The two assumptions cleanly separate past-task test samples into "correctly classified" (Assumption 1) and "misclassified into current task" (Assumption 2), enabling tailored correction strategies. Table 5 validates these assumptions empirically, showing 88.4% and 71.9% accuracy respectively (Section 3.3, Table 5).

- **Compelling motivating experiments**: Figure 1 isolates classifier bias by fixing the backbone and comparing shared vs. independent classifiers; Figure 2 visualizes the skewed distribution of misclassifications toward recent tasks. These provide clear, quantitative evidence for the problem ARC addresses (Section 1, Figures 1–2).

- **Superiority over generic TTA**: Table 3 shows ARC consistently outperforms TENT, with average performance gaps of 1.8% on Split CIFAR-100 Inc10 and 2.9% on Split ImageNet-R Inc20, and particularly larger forgetting reductions. This demonstrates the value of ARC's CL-specific selective adaptation over generic TTA (Section 4.2, Table 3).

- **Plug-and-play design**: ARC operates entirely at test time (Algorithm 1), making it trivially composable with any existing CL method without training modifications—a significant practical advantage.

## Weaknesses

### Fatal
None.

### Major

- **Evaluation protocol conflation of test-set adaptation with genuine de-biasing**: Adaptive Retention updates the classifier on test data using pseudo-labels (Algorithm 1, lines 4–5), meaning the model is modified during the very evaluation phase where accuracy is measured. The reported accuracy therefore reflects both the original model's performance AND the benefit of having adapted to test data. While this is inherent to the TTA paradigm ARC operates in, the paper does not verify whether the improvements would hold on a held-out set not used for classifier updates. A simple 50/50 split protocol (use one half for Adaptive Retention updates, report accuracy on the other half) would disentangle genuine de-biasing from test-set adaptation. The TENT comparison (Table 3) partially addresses this by showing ARC outperforms a generic TTA method, but TENT is only one baseline—more TTA methods (e.g., EATA, CoTTA, SAR) would strengthen the claim that ARC's CL-specific design, rather than generic test-time adaptation, drives the improvements. This concern does not invalidate the results but limits confidence that the gains are as large as reported.

- **Limited isolation of ARC's CL-specific contribution from generic test-time training**: Related to the above, the paper positions ARC as a CL-specific method, but the Adaptive Retention component (entropy minimization + pseudo-label cross-entropy on confident predictions) is essentially selective test-time training. The paper would benefit from a more thorough disentangling: for instance, applying only entropy minimization (the non-task-specific component) to all test samples and comparing with ARC's selective approach would isolate the contribution of OTD and Assumption-based selection. The L2P ablation in Figure 4a is revealing in this regard: entropy minimization alone achieves 86.6% vs. the combined loss at 86.2%, suggesting pseudo-labels may not always add value and that the generic TTA component may be doing most of the work for some methods.

### Minor

- **"Primary source of forgetting" claim is overclaimed**: Section 3.2 states "the key issue... lies in the bias introduced by the classifier" and that representation layers "are adept at retaining knowledge." The evidence (Figure 3) shows linear probing recovers high accuracy after fine-tuning, but linear probing re-trains the classifier on past-task data with ground-truth labels, which removes classifier bias by construction. This shows features remain discriminative, not that classifier bias is the *primary* bottleneck. The representation and classifier could shift in coupled ways that linear probing masks. A more careful phrasing ("a major contributing factor" rather than "the key issue") would be more appropriate (Section 3.2).

- **Pseudo-label error accumulation is unanalyzed**: Table 5 shows Assumption 1 accuracy ranges from 81.6% (DualPrompt) to 92.8% (CodaPrompt), meaning 7–18% of samples receive incorrect pseudo-labels that trigger gradient updates on the classifier. The paper employs entropy minimization to mitigate noise (Section 3.4.1) but does not analyze whether error accumulation compounds over sequential updates, whether confirmation bias emerges, or under what conditions Adaptive Retention degrades performance. The CodaPrompt Split CIFAR-100 Inc5 case (Table 1: forgetting increases from 5.7% to 5.8%) hints at potential harm, though the effect is marginal.

- **Hyperparameter sensitivity analysis is limited**: Figure 5 tests β and γ on only 4 of 8 methods and only on Split CIFAR-100. The method's robustness on ImageNet-R (where the distribution is more diverse) or with prompt-based methods is not verified (Section 4.4).

- **No variance reported**: Results are single-run without standard deviation. Some improvements are modest (e.g., +0.7 for SLCA on Split CIFAR-100 Inc10, +0.9 for DER on 5-dataset), making statistical significance uncertain.

### Trivial
None.

## Nice-to-Haves

- Ablation on the number of gradient steps per sample in Adaptive Retention to study the tradeoff between adaptation speed and error accumulation.
- Per-task accuracy breakdown showing which tasks benefit most from ARC, rather than only average accuracy.
- Comparison with more recent TTA methods (EATA, SAR, CoTTA) beyond TENT.
- Analysis of TSS temperature scaling robustness across varying numbers of tasks.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "memory-free" claim is misleading because ARC uses test samples as a de facto replay buffer**: This mischaracterizes ARC. ARC does not *store* any data; it processes each test sample once with a single gradient update and moves on. There is no replay buffer. The method is genuinely memory-free in the CL sense (no stored exemplars during training). The test-time operation is a feature, not a bug.

- **Harsh critic: TSS requires knowing task boundaries and incremental step size**: The critic themselves acknowledge this is standard in class-incremental learning. This is not a weakness of the method.

- **Harsh critic: Table 4 shows Adaptive Correction contributes little for prompt-based methods, questioning its centrality**: The paper already discusses this finding transparently and explains the cause (few samples satisfy Assumption 2 for prompt-based methods). This is honest reporting, not a hidden weakness.

- **Harsh critic: Single-run results without variance**: While true, this is standard practice in the CL field for large-scale benchmark evaluations. Flagged as minor rather than major.

- **Strength finder: "Strong empirical finding that pretrained model representations resist catastrophic forgetting"**: While Figure 3 is a useful contribution, it supports a slightly overclaimed conclusion as noted in the Minor weaknesses above. Downgraded accordingly.

## Novel Insights

The paper's most insightful observation is that standard CL evaluation protocols naturally provide access to past-task data at inference time, yet this data has been systematically ignored by the community. This reframes the "memory-free" constraint: the issue is not that past-task data is unavailable, but that it arrives unlabeled and without task identity. ARC's OTD framework converts this overlooked resource into a practical de-biasing mechanism. The finding that even generic TTA (TENT) helps CL methods—but that CL-specific selective adaptation helps more—suggests a productive intersection between the TTA and CL communities that has been underexplored.

## Suggestions

- Run a held-out evaluation: split each task's test set 50/50, use one half for Adaptive Retention updates, and report accuracy on the other half. This single experiment would address the most significant weakness.
- Add a "pure entropy minimization on all test samples" baseline to isolate the contribution of OTD-based selection from generic TTA.
- Soften the "primary bottleneck" claim in Section 3.2 to acknowledge that the evidence supports "a major contributing factor" rather than "the key issue."

## Score and Decision

Calibration anchors compared:

| Paper | Avg Score | Decision | Relation to ARC |
|-------|-----------|----------|-----------------|
| SD-LoRA (5U1rlpX68A) | 7.5 | Oral | Stronger theoretical depth, cleaner contribution |
| Entropy not Enough for TTA (9w3iw8wDuE) | 7.0 | Spotlight | Deeper analysis of TTA bias, stronger baselines |
| Controlling Forgetting with TTD (fRNDDFkPiv) | 6.75 | Reject | Very similar concept; ARC has clearer methodology and broader experiments, partially addresses fairness concern with TENT comparison |
| SlimTTT (7iuFxx9Ccx) | 6.0 | Reject | Simpler TTA approach, less comprehensive |
| Analytic CTTA (eXrUdcxfCw) | 5.0 | Reject | Minor improvements, limited novelty |
| PRO pseudo-label TTA (KZZbdJ4wff) | 3.75 | Reject | Just combines existing methods, no depth |
| How OOD important is (10fsmnw6aD) | 2.5 | Reject | Poor presentation, unclear contributions |

ARC is clearly stronger than the low-scoring anchors (2.5–5.0). It is comparable to or slightly better than fRNDDFkPiv (6.75) — ARC has clearer methodology, broader experiments, and the TENT comparison partially addresses the fairness concern that sank fRNDDFkPiv. However, ARC is weaker than the 7.0+ papers, which have either deeper theoretical analysis or more rigorous evaluation methodology. The evaluation protocol concern is the primary factor keeping ARC below the acceptance threshold used for the high-scoring anchors.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>