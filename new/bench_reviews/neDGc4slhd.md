## Summary

This paper introduces Betti curve similarity (BCS) as a tool for comparing the global functional structure of deep neural networks using topological data analysis. The authors train four CNN architectures (LeNetExt, AlexNet, VGG-16, ResNet-18) on 30 disjoint 10-class subsets of ImageNet, construct functional graphs from layer activations via k-means++ reduction and Vietoris–Rips persistent homology, and use the infinity norm between Betti curves to quantify similarity across architectures, datasets, and training epochs. The study is large-scale (120 model instances) and reproducible, with code and seeds provided.

## Strengths

- **Novel application of TDA to cross-model comparison.** To the authors’ knowledge, this is the first use of Betti curve similarity to compare DNN functional graphs across both datasets and training epochs (Section 2.5). The full pipeline—from activation extraction through k-means++ reduction to persistent homology and BCS—is operationalized end-to-end.
- **Evidence that BCS captures structure beyond accuracy.** In Section 3.2, the paper presents case studies (subsets 11 and 27) where BCS diverges from classification performance: ResNet-18 and VGG-16 achieve comparable accuracy on subset 11 yet exhibit low topological similarity, while on subset 27 LeNetExt is topologically dissimilar from deeper CNNs despite distinct accuracy rankings. This suggests BCS may encode representational information not redundant with task performance.
- **Reproducible and large-scale experimental design.** The study trains 120 model instances and provides random seeds, software versions, hardware details, and code links (Sections 2.1–2.2). The computational investment is substantial.

## Weaknesses

### Fatal
None.

### Major
- **Severely suboptimal training protocol undermines the empirical foundation.** All four architectures are trained with identical hyperparameters: Adam optimizer, learning rate 0.001, weight decay 0.0005, no learning rate scheduling, and only 60 epochs (Section 2.2). Modern architectures such as ResNet-18 and VGG-16 are known to require SGD with momentum, careful learning rate tuning, and longer training to converge on ImageNet-scale data. The result is catastrophically low average test accuracy: Figure 2 shows all models plateauing at roughly 35–45% on 10-way classification, whereas a properly trained ResNet-18 on 64×64 ImageNet subsets should achieve far higher performance. While the paper does not claim state-of-the-art convergence, analyzing “global structure” in networks that are stuck in poor optima means the topological signatures may not reflect the functional behavior these architectures typically exhibit. The core methodological idea remains valid, but confidence in the empirical observations is substantially weakened.
- **No baseline comparison against standard representation similarity metrics.** The paper proposes BCS as a similarity measure (Equation 7) but never compares it to established metrics such as CKA, SVCCA, or even simple matrix norms applied to the same activations. Without this, the reader cannot assess whether the topological machinery adds value over simpler, widely used methods. For a paper whose contribution is introducing a new similarity tool, this omission is a significant gap.

### Minor
- **Betti curve similarity lacks theoretical or empirical justification.** The infinity norm between Betti curves is introduced without motivation (Section 2.5). Betti curves are unstable summaries of persistence diagrams, and the paper provides no argument for why the infinity norm should be preferred over standard topological distances (bottleneck, Wasserstein) or other p-norms. The measure reads as a descriptive statistic whose functional relevance is assumed rather than established.
- **No quantification of variance or stability.** The heatmaps and line plots report average similarities with no indication of variance across the 30 subsets or across random initializations (Figures 4–9). If BCS is highly variable, the interpretability of the averaged patterns collapses.
- **k-means++ approximation is unvalidated.** The paper honestly reports that silhouette scores indicate “poorly separated” clusters (Section 2.3) and argues that global structure is nonetheless preserved, citing Corneanu et al. (2019). However, no experiment is provided to show that persistent homology computed on the 1000-cluster reduced set approximates the true PH of the full activation data.

### Trivial
- **Pseudometric nature of $d_\rho$ is honestly disclosed.** The paper notes that $d_\rho$ violates positivity (Section 2.4). While Vietoris–Rips complexes technically assume a metric for certain guarantees, persistent homology can still be computed on dissimilarity matrices, and the authors’ transparency here is appropriate.

## Nice-to-Haves
- Retrain with architecture-specific hyperparameters and learning rate schedules to obtain properly converged models, then recompute all topological analyses.
- Compare BCS against CKA, SVCCA, and simple activation correlation metrics on identical activations to validate added value.
- Add error bars or variance heatmaps showing BCS standard deviation across subsets and random seeds.
- For a small tractable layer, compute PH on the full activation set and compare to the k-means-reduced version to justify the approximation.
- Connect BCS to a downstream functional quantity (e.g., transfer learning performance or robustness) to demonstrate practical utility.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Cross-model comparisons are confounded by data distribution”**: This criticism mischaracterizes the paper’s core architecture comparisons. Section 3.2 explicitly compares different models on the *same* subset (e.g., ResNet-18 vs. VGG-16 on subset 11, and all four models on subset 27; Figures 6–9), which controls for data identity. While some aggregate heatmaps (e.g., Figure 4) show cross-subset patterns, the key claims about architectural discrimination are grounded in same-subset analysis.
- **“The pseudometric invalidates Vietoris–Rips interpretation”**: The paper honestly discloses that $d_\rho$ is a pseudometric, and VR complexes can be computed on general dissimilarity matrices. The theoretical guarantees differ from the metric case, but this does not render the computation meaningless.
- **“The training protocol is invalid / methodologically indefensible”**: While the identical hyperparameter choice is suboptimal for deeper architectures, the paper explicitly frames it as a controlled design to isolate architectural effects. The flaw is poor convergence, not methodological cheating or incoherence. The networks do learn (accuracy rises from ~10% to ~40%), so “invalid” overstates the case.

## Novel Insights

The observation that topological similarity can diverge from task accuracy—most clearly shown in the subset-11 and subset-27 case studies—is genuinely interesting and suggests that persistent homology may capture representational structure invisible to standard performance metrics. If this finding survives on properly trained networks and against appropriate baselines, it could motivate a useful new diagnostic tool for deep learning.

## Suggestions

1. **Retrain with valid hyperparameters.** Use SGD with momentum, proper learning rate scheduling, and longer training for ResNet-18 and VGG-16. Recompute all PH and BCS analyses.
2. **Add baseline similarity metrics.** Compute CKA and SVCCA on the same reduced activations and report head-to-head comparisons.
3. **Quantify stability.** Report BCS variance across at least three random initializations per architecture-subset pair.
4. **Justify or replace the infinity norm.** Either provide theoretical/empirical motivation for the infinity norm on Betti curves, or benchmark against bottleneck and Wasserstein distances.

## Score and Decision

**Calibration papers compared:**
- `/home/wg25r/review_agent/human_reviews/mYgoNEsUDi.md` (avg 6.33, Accept): A TDA paper with theoretical stability guarantees, extensive baseline comparisons, and clear performance gains. The paper under review lacks these strengths and scores below it.
- `/home/wg25r/review_agent/human_reviews/cMQeDPwSrB.md` (avg 5.20, Reject): A metric paper with sound experiments but no baseline comparisons. The paper under review shares the missing-baseline flaw but adds a suboptimal training protocol, placing it below this anchor.
- `/home/wg25r/review_agent/human_reviews/JHE4w8q2G2.md` (avg 4.50, Reject): An empirical deep learning paper with unfair comparisons, missing baselines, and no statistical significance. Comparable in severity to the training-protocol and missing-baseline issues here.
- `/home/wg25r/review_agent/human_reviews/lf8QQ2KMgv.md` (avg 3.75, Reject): A paper with an actually invalid training protocol (test-set-guided convergence). The current paper’s protocol is merely suboptimal, not invalid, so it should score above this anchor.

The paper under review sits between the 3.75 and 5.20 anchors. Its novel TDA application and large-scale execution lift it above the truly flawed low end, but the suboptimal training protocol and complete absence of baseline similarity comparisons keep it well below the acceptance threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>