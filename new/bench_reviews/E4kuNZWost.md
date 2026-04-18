Now I have enough calibration context. Let me write the final consolidated review.

## Summary
TULiP proposes a post-hoc OOD detection method grounded in the linearized (lazy) training dynamics of neural networks. The key theoretical result (Theorem 3.1) bounds the deviation between perturbed and unperturbed training trajectories in terms of NTK-based distances between test and training data, motivating an uncertainty score estimated via weight perturbations without requiring training data. The method achieves strong near-OOD detection performance on OpenOOD benchmarks.

## Strengths
- **Novel theoretical framework**: The connection between epistemic uncertainty and training-dynamics perturbations via the NTK (Theorem 3.1, Proposition 3.3) is original and provides a principled geometric interpretation of OOD detection as distance in the gradient embedding space. The derivation from perturb-then-train dynamics to a computable bound is technically nontrivial and a genuine contribution to the theory of epistemic uncertainty.
- **Strong near-OOD performance**: TULiP achieves top-1 or top-2 AUROC on near-OOD across all benchmarks in Table 1 (e.g., 89.67/92.55 on CIFAR-10, 83.84/91.03 on ImageNet-200), consistent with the theoretical prediction from Eq. 8 that the method should excel when test points are reasonably close to the training distribution.
- **Practical efficiency and versatility**: Algorithm 1 avoids explicit Jacobian computation via first-order approximations (Eq. 13) and Hutchinson's trace estimator (Prop. 4.1), requiring only O(M) forward passes. The method can be combined with existing logit-based methods (TULiP+GEN) and naturally extends beyond classification.
- **Careful empirical analysis**: The ablation study (Fig. 4) clearly reveals the near/far OOD trade-off as a function of hyperparameters. Figure 1 provides empirical grounding for the layer-wise scaling and closeness assumption.

## Weaknesses

### Fatal
None.

### Major

- **Substantial theory-practice gap undermines the "theoretically-driven" framing**: The central claim that TULiP is "theoretically driven" by training-dynamics analysis is weakened because the implemented algorithm departs significantly from the derived bound at multiple steps: (1) The layer-wise scaling (Eq. 12), which is critical for practical performance and without which "our method failed to achieve consistent performance," is acknowledged as "highly heuristic" and is not derived from the theory. (2) The constant K in Lemma 3.2 is replaced by a tunable hyperparameter λ selected on validation data, breaking the bound interpretation. (3) The term E_x[Θ(x,x)] is dropped entirely. (4) Prop. 4.1 requires ε → 0, but ε is tuned over {0.1, 0.5, 1.5, 2.0}. These departures are individually acknowledged, but their cumulative effect is that the theory motivates rather than constrains the algorithm. The "theoretically-driven" framing should be significantly softened — the theory provides inspiration, not a justification for the specific score computed in practice. This matters because it is the paper's primary distinguishing claim over existing post-hoc heuristics.

- **Theory relies on assumptions violated in the empirical setting**: Assumptions A1–A4 (lazy regime with constant NTK, bounded Jacobians, Lipschitz loss, near-perfect convergence, closeness condition) are stated for the theoretical framework but are not satisfied by the ResNet-18/50 models trained with SGD-momentum on CIFAR/ImageNet. The authors themselves note "Lazy training often fails to capture the full characteristics of practically trained neural networks" and "significant changes in the empirical NTK" (Sec. 4.1). The synthetic validation (Fig. 2) confirms the bound only in the infinite-width lazy regime. There is no analysis of how or why the bound remains meaningful when the NTK varies substantially. The paper would be significantly strengthened by validating whether the bound (or its key qualitative properties) correlates with actual ensemble variance on practical (non-lazy) networks.

### Minor

- **Near/far OOD trade-off limits generality**: TULiP consistently underperforms on far-OOD (e.g., ImageNet-1K far-AUROC 88.03 vs. ASH's 95.74). While this is consistent with the theory and acknowledged by authors, a method whose performance depends heavily on unknown OOD distance raises practical deployment concerns. The paper could clarify the expected operating regime more prominently.

- **Limited architecture diversity**: Evaluation is primarily on ResNets, with a small additional study on MobileNet-V3, VGG-16, and RegNet-Y-16GF (Fig. 3). Modern architectures like Vision Transformers are only briefly mentioned in the conclusion/appendix as future work, leaving the generality claim under-supported.

- **Non-classification generality claim is speculative**: The abstract states the method "is not limited to classification problems," but no non-classification experiment is provided. This claim should either be demonstrated or qualified.

- **Computational overhead not systematically compared**: TULiP requires M=10 forward passes per input. While the paper mentions it is 3× faster than ViM, no systematic wall-clock comparison with single-pass methods (MLS, EBO) or other perturbation methods is provided.

### Trivial
- None.

## Nice-to-Haves
- Validate the theoretical bound on practical (non-lazy) networks, e.g., by computing ensemble variance for small models and checking correlation with TULiP's score.
- Evaluate on Vision Transformers to test architectural generality.
- Report a non-classification task (e.g., regression) to support the generality claim.

## Removed Points

- **"The method is not truly post-hoc because it requires hyperparameter tuning on validation data"**: Hyperparameter tuning on validation sets is standard practice in OOD detection (all baseline methods in Table 1 also use validation tuning). The "post-hoc" claim specifically means no access to training data or training process — this is distinct from validation-set hyperparameter selection. *Weakened* — the concern about needing validation data that resembles the OOD type (near vs. far) has some merit but is minor and common to many methods.

- **Missing comparisons with deep ensembles and Laplace approximation methods on large-scale benchmarks**: Deep ensembles require retraining (5+ models), making them a fundamentally different category from post-hoc methods. Laplace approximation methods have their own significant theory-practice gaps for deep networks and are not standard OOD baselines on ImageNet. While comparing against them would strengthen the paper, their absence is not a critical flaw given the post-hoc framing. *Moved to Nice-to-Haves.*

- **Missing comparison with ReAct and RankFeat**: These are mentioned as related plug-and-play methods in Sec. 2 but absent from tables. This is a minor omission — the included baselines (ODIN, EBO, MLS, ASH, GEN) cover the main post-hoc methods, and ASH is the strongest among them. *Removed as a minor concern about completeness, not a methodological flaw.*

- **Assumption A3 (uniform perturbation bound over all R^d) is extremely strong**: While A3 is indeed a strong assumption, the perturbation Δf is hypothetical (never actually applied) and serves only to derive the bound. The bound's role is to motivate the practical score, not to provide a tight guarantee. *Weakened* — noting it as an assumption gap but not a fatal flaw since it motivates rather than constrains the method.

- **Assumption A4 (near-perfect convergence) is strong**: Overparameterized networks do achieve near-zero training loss in practice for cross-entropy with appropriate training. The authors cite relevant literature (Zhang et al., 2017; Du et al., 2019). This is a standard assumption in the NTK literature. *Removed as a standalone weakness since it is adequately motivated.*

## Novel Insights
The most interesting insight is the inherent tension revealed by the paper's own results: TULiP's theory predicts strength on near-OOD (Eq. 8 connects better detection to proximity in NTK space), and this is precisely what the experiments confirm. This suggests that the NTK-proximity intuition, even when the formal assumptions are violated, carries qualitative predictive power. However, the flip side — that the same framework struggles with far-OOD — reveals a fundamental limitation of NTK-based uncertainty: the framework needs training-set-adjacent inputs to produce discriminative scores, and its signal degrades precisely where it becomes most important to detect genuinely novel inputs.

## Suggestions
- Soften the "theoretically-driven" framing to "theoretically-inspired" and explicitly acknowledge the theory-practice gap in the method section. The paper would be stronger by being transparent about where theory drives design vs. where heuristic choices are made.
- Provide a correlation analysis between the TULiP score and deep ensemble variance on practical networks (even small ones) to validate whether the method captures genuine epistemic uncertainty.
- Report wall-clock time per image alongside all baselines in a dedicated table.

**Evaluation on key axes:**
- **Originality**: High. The NTK-based fluctuation bound for OOD detection is novel and provides a fresh perspective.
- **Importance of research question**: High. Post-hoc OOD detection with theoretical grounding is important.
- **Claims well-supported**: Partially. The theoretical claims are correct within their regime, but the link to the implemented method is loose. Empirical claims are well-supported for near-OOD.
- **Soundness of experiments**: Good. OpenOOD is a standard benchmark, baselines are appropriate, and 3-run averaging is reported. However, no significance tests and no comparison on ViTs.
- **Clarity**: Good overall, but Section 4's connection to Section 3 could be made more transparent about where heuristic departures occur.
- **Value to community**: Moderate-to-high. The theoretical framework opens a promising direction, and the method is practical, but the theory-practice gap limits immediate trust in the theoretical guarantees.

**Calibration**: Compared to similar papers at this venue: The SCALE/ASH OOD paper (scores 5,8,6,6, avg ~6.25, accepted poster) had simpler theory but consistent empirical gains. The NTK theory-practice disconnect paper (scores 5,5,8,6, avg ~6, borderline/reject) explicitly demonstrated that NTK assumptions fail in practice — directly relevant to TULiP's theoretical foundation. The implicit functional Bayesian DL paper with shaky NTK justification (scores 3,5,3,5, avg ~4, rejected) is a cautionary parallel. TULiP sits between these: it has genuine theoretical novelty and strong near-OOD results, but its theory-practice gap is significant and honestly acknowledged rather than hidden. The main contribution remains valuable even if the theoretical framing is overstated.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>