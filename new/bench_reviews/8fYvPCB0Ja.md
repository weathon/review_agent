Now let me search for comparable papers to calibrate my score.Now I have enough to write the final review. Let me synthesize.

---

## Summary
FairDD proposes a fair dataset distillation (FDD) framework that reformulates matching-based dataset distillation objectives by synchronously aligning synthetic samples to each protected-attribute (PA) group equally, rather than to the whole distribution dominated by the majority group. The method requires no architectural modifications to existing DD approaches and provides theoretical analysis showing that (1) the synthetic representation converges to the arithmetic mean of PA group centers rather than the majority-weighted mean, and (2) the FairDD objective upper-bounds the vanilla DD objective under convex distance functions, guaranteeing at least equal distributional coverage. Extensive empirical results across four DD paradigms, six datasets, and multiple architectures demonstrate large fairness improvements over vanilla DD methods with simultaneously maintained or improved accuracy.

---

## Strengths

- **First to formally address PA fairness in dataset distillation.** The paper makes a compelling and original observation that vanilla DD methods not only fail to mitigate PA bias but actively amplify it. The DEO values of 100.0 (e.g., DM on C-MNIST FG IPC=10) vs. ~10 for the whole dataset quantify this clearly.

- **Simple, elegant, and broadly applicable method.** The core modification—decomposing the single alignment target into PA-wise sub-targets and weighting them equally (Eq. 7)—requires no new modules or hyperparameters and is verifiably applicable to both DM and GM paradigms. This plug-in design has high practical value.

- **Comprehensive empirical evaluation.** Results span 5 datasets (FG/BG synthetic variants, CIFAR10-S, and CelebA), 4 DD methods, 3 IPC settings, and cross-architecture generalization (ConvNet, AlexNet, VGG11, ResNet18). This breadth meaningfully supports the "versatile FDD approach" claim.

- **Dramatic and consistent fairness gains.** Improvements are not marginal: DEO_M drops from 100.0 → 17.04 (DM+FairDD on C-MNIST FG, IPC=10), and the accuracy simultaneously improves from 25.01% → 94.61%. The pattern holds across all DD methods tested. The t-SNE visualizations (Fig. 5) are consistent with the claimed mechanism.

- **Meaningful theoretical support.** Theorem 4.1 rigorously shows the synthetic expectation is independent of group sample ratio at optimum. Theorem 4.2 establishes FairDD as an upper bound of the vanilla objective under convex distance functions, providing a valid coverage guarantee. These are true mathematical results, not heuristic arguments.

---

## Weaknesses

### Fatal
*None. The paper's core claims are credible and supported by evidence.*

### Major

- **Missing PA-aware balancing baseline.** FairDD explicitly uses PA labels to balance matching across groups. The most natural alternative explanation for the reported gains—especially the dramatic accuracy improvements—is that simply subsampling the original dataset to balance PA groups before applying vanilla DD would achieve similar results. This comparison is absent. Without it, the paper cannot establish that synchronized matching per se is the key contribution versus generic use of group labels for balancing supervision. This is the most critical experimental gap.

- **Theoretical "guarantees" are moderately overstated relative to what the math shows.** Theorem 4.1 establishes a property of the optimization surrogate's optimum in feature space (the mean of PA group signal means), not a downstream equalized odds guarantee. Theorem 4.2 establishes an upper-bound relationship between two objectives, not that the distilled set achieves broader distributional coverage in any task-performance sense. The abstract and introduction repeatedly claim "theoretical guarantee that FairDD significantly improves fairness... without sacrificing classification accuracy"—this wording implies stronger end-to-end guarantees than what the proofs actually deliver. The theorems are real and meaningful; they should just be described more precisely. The Harsh Critic correctly identifies this issue.

- **Evaluation heavily concentrated on synthetic benchmarks.** Four of five datasets use artificially injected color or grayscale bias, with test sets engineered to be PA-balanced. These settings are neatly aligned with the proposed mechanism (discrete binary PA, visually salient bias, easy partitioning). CelebA is the only genuinely real-world dataset. The paper's broad claims about applicability "regardless of PA imbalance in the original data" are insufficiently supported when real-world bias is noisier, subtler, and multi-dimensional. The neutral reviewer and Spark both flag this correctly.

### Minor

- **No variance reporting across runs.** Dataset distillation is known to be high-variance. Main tables report point estimates only, making it hard to assess whether modest improvements (particularly on CelebA) are statistically meaningful. This is especially consequential for "consistent superiority" claims.

- **TM exclusion lacks empirical justification.** The paper justifies excluding Trajectory Matching with a brief statement about overfitting on minority groups during trajectory generation (Sec. 1). Since TM is a dominant DD paradigm, even a small-scale demonstration of this failure mode would strengthen the paper's scope justification.

- **Dependency on PA labels is a real limitation with no exploration.** FairDD requires ground-truth PA annotations at distillation time. The limitations section acknowledges this but offers only future-work language (pseudo-labels via self-supervised learning). Even a brief experiment using clustering or a simple PA classifier as proxy would make this discussion meaningful.

### Trivial

- The conceptual gap between the optimization target (mean of PA group feature expectations) and the evaluation criterion (equalized odds) is not explicitly acknowledged. A brief clarifying sentence would help.

---

## Nice-to-Haves

- A comparison with PA-balanced random subsampling followed by vanilla DD would directly test whether the synchronized matching mechanism adds value over simpler group rebalancing.
- Per-PA-group accuracy breakdown in addition to aggregate DEO metrics would clarify whether FairDD lifts minority groups without harming majority groups.
- Report of compute overhead (currently in Appendix C) should be promoted to the main paper, as FairDD multiplies the matching target count by |A|, which is relevant for scalability.
- At least one additional real-world dataset (e.g., UTKFace with gender/race PA) would substantiate the generality claim.
- Empirical demonstration of TM failure would complete the scope justification.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "The move from Eq. (4)/(6) to the statement that minority groups experience higher loss during testing is not established by the derivation."** — Partially valid, but the paper's claim (Sec. 3, last paragraph) refers to training dynamics and optimization bias, not strictly to test-time loss. The text says "minority groups experience higher loss during testing, which limits the model to represent them accurately" — this is a reasonable inference from the optimization analysis even if not formally proved. Not a strong enough standalone criticism to keep.

- **Harsh Critic: "The unified view in Eq. (3) abstracts away important differences among DM/DC/IDC/DREAM."** — The paper introduces this as a "unified perspective" for analysis, not as a universal theorem. Each method's specific form differs; the abstraction is pedagogically useful. This is a scope creep criticism, removed.

- **Harsh Critic claim about the vanilla DD baseline not being "optimally tuned."** — The paper uses official implementations for all baselines. This is a reproducibility concern that falls under removing implementation nitpicks. Removed.

- **Human Finder: "Reliance on a single fairness metric (equalized odds)."** — The paper uses both DEO_M and DEO_A, covering worst-case and average group disparities. The concern about impossibility theorems (demographic parity vs. EO tradeoffs) is valid in principle but is outside the paper's stated scope and would apply to virtually all fairness papers. Evaluated as scope creep/generic criticism.

- **Spark: "No experiments with multiple or intersecting protected attributes."** — A valid limitation, but it is outside the paper's stated scope. The paper proposes a first FDD framework with binary PA; extensions to multi-PA settings are a natural next step, not a prerequisite for the current contribution.

---

## Novel Insights

The most genuinely novel insight from this paper—beyond its methodological contribution—is the diagnostic finding that vanilla DD systematically collapses the condensed dataset toward majority-group feature means, and that this collapse simultaneously degrades both fairness *and* accuracy when the test distribution is PA-balanced. This coupling between fairness failure and accuracy failure is underappreciated: it implies that vanilla DD is not merely unfair but actively degenerate on biased datasets in a way that synchronized matching completely resolves. The dramatic accuracy improvements (e.g., 25% → 94% for DM on C-MNIST FG) are not an artifact—they reflect a fundamental representational failure of vanilla DD under PA imbalance, exposed clearly for the first time.

---

## Suggestions

1. Add a PA-balanced subsampling baseline (subsample original data to equalize PA groups per class, then run vanilla DD) as a direct ablation to isolate the contribution of synchronized matching from simpler group-balancing.
2. Revise Theorem 4.1 and 4.2 descriptions to state precisely what is proved (properties of the surrogate objective optimum and upper-bound relationships) without overstating downstream fairness guarantees.
3. Add per-PA-group accuracy bars to the analysis section to demonstrate that improvements come from lifting minority groups rather than degrading majority groups.
4. Include at least one additional real-world naturally biased dataset to broaden the generalization claim.
5. Promote computation overhead figures to the main paper.

---

## Score and Decision

**Calibration anchors:**

- *L5yq5KsnEZ* ("Mitigating Bias in DD", Withdrawn/Reject, scores 5,3,5,5, avg ~4.5): Same problem area, but weaker—only 2 DD methods, no principled framework, no real-world dataset. FairDD is clearly superior.
- *MIuimtOu0T* ("BIRD: Fair Knowledge Distillation", Reject, scores 6,6,5,5,6, avg ~5.6): Fair distillation in a different paradigm (knowledge distillation). Comparable rigor but less original (prior fairness-in-KD work exists), while FairDD is the first fairness-in-DD work.
- *rTBL8OhdhH* ("Lossless DD via Difficulty-Aligned TM", Accept poster, scores 8,6,8,6, avg ~7): Accepted DD paper with strong theory + experiments. FairDD's theory is weaker and experiments lack key baselines.
- *h57gkDO2Yg* ("Self-Supervised DD for Transfer Learning", Accept poster, scores 6,8,6,5,6, avg ~6.2): Accepted, novel problem in DD domain with solid experiments. FairDD is comparable in novelty and empirical scope.

**Positioning:** FairDD is substantially above L5yq5KsnEZ (its nearest topical neighbor), broadly comparable to BIRD in rigor but more original (first in area), and below the strong accepted DD papers in theoretical depth and experimental completeness. The key missing baseline (PA-balanced subsampling) and modest theoretical overclaiming prevent a confident accept, but the genuine novelty, strong empirics, and practical importance support borderline acceptance.

**Evaluation summary:**
- *Originality*: High — first explicit FDD framework
- *Importance*: High — fairness in compressed datasets is practically significant
- *Claims well supported*: Mostly — strong empirics, but missing a key ablation baseline and overclaimed theory
- *Soundness of experiments*: Good — comprehensive, but lacks variance and key baselines
- *Clarity of writing*: Good — well-organized and readable
- *Value to research community*: High — opens a new research direction with a clear, replicable method

**Score: 5.5** — Borderline, leaning toward weak accept. The contribution is genuine and the empirical evidence is compelling; the paper is clearly publishable with the missing baseline added and theoretical claims recalibrated.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>