Now I have enough calibration data. Let me synthesize the review. The key calibration points:

- Papers on NTK theory gaps with practical implications (GqI4fTVUXC - scores 5,5,8,6; XgAKt7rbXk - scores 3,5,3,3) tend to be in the 3-6 range, with lower scores when theory-practice gaps are severe.
- Papers on OOD detection with practical strong results (RDSTjtnqCg - scores 5,8,6,6) get scores around 5-8.
- The ifBDL paper with shaky NTK justification (ohHtdp3jDi) scored 3,5,3,5 (rejected).

Now let me carefully cross-check all the criticisms against the paper and write my review.

Key issues to verify:
1. Theory-practice gap between NTK/lazy regime and practical implementation
2. Heuristic implementation deviating from theoretical bound
3. No validation against ensemble/Bayesian uncertainty on real models
4. Layer-wise scaling being heuristic
5. NTK divergence under cross-entropy
6. Hyperparameter sensitivity and near/far-OOD tradeoff
7. Transformer/other architecture evaluation

Let me carefully go through each point that the harsh critic raised.

For point 1 (theory-practice gap): This is substantively correct. The paper derives everything under lazy/NTK regime, but then uses ResNets with SGD where Fig 1a shows the NTK changes significantly. The layer-wise scaling is admitted as heuristic. This is a real issue.

For point 2 (Prop. 3.3 and Algo 1 fragility): This is also correct. Multiple heuristic steps separate the theory from implementation (θ_{t_s} → 0, dropping ℰ_x[Θ(x,x)], replacing K with λ, max(S,0) clamping). The bound is no longer a bound at that point.

For point 3 (no ensemble validation): Correct - no comparison with deep ensemble variance on real models.

For point 4 (layer-wise scaling heuristic): Correct - admitted as heuristic, only one plot of justification.

However, I should also note the paper's genuine strengths:
- Novel theoretical framework connecting training dynamics to OOD detection
- The synthetic validation (Fig. 2) is nice
- Strong empirical results on OpenOOD, especially near-OOD
- Computationally efficient (O(M) forward passes)
- Works across architectures without training data access

The human finder raised NTK divergence under cross-entropy - let me check if this is relevant. The paper uses cross-entropy loss for classification. The NTK divergence paper showed that under cross-entropy, parameters diverge, which is problematic for the lazy regime assumption. However, the paper uses convergence assumption A4 and notes it applies through overparameterization. But the NTK divergence result directly challenges this. This is a valid concern.

Let me now write the final review.</think>Now I have sufficient calibration context. Let me carefully verify all claims against the paper text and write the meta-review.

## Summary

TULiP proposes a post-hoc OOD detection method grounded in the theory of linearized (lazy/NTK) training dynamics. The key idea is to bound the effect of hypothetical perturbations applied before convergence (Thm. 3.1), then estimate this bound at test time via weight perturbations without accessing training data (Prop. 3.3, Algo. 1), producing surrogate posterior samples whose entropy yields an uncertainty score. The method achieves state-of-the-art near-OOD detection on the OpenOOD benchmark across multiple architectures.

## Strengths

- **Novel theoretical framing.** Deriving an OOD uncertainty bound from perturbations to linearized training dynamics (Thm. 3.1, Eq. 5) is a genuinely novel contribution that connects test-time uncertainty to the training process in a principled way. The bound's dependence on NTK-distance from the training set (Eq. 6) is intuitive and illuminating.

- **Efficient practical implementation.** The algorithm avoids explicit Jacobian computation via Hutchinson's trace estimator and first-order finite differences (Props. 4.1, Eq. 13), requiring only O(M) forward passes. This is a clever engineering choice that makes the method scalable.

- **Strong empirical performance on near-OOD.** TULiP consistently achieves top-1 or top-2 AUROC for near-OOD across CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K (Table 1), outperforming methods like ViM that require training data access. The synthetic validation (Fig. 2) is clean and supports the bound in its own regime.

- **Generality beyond softmax-based scores.** The method constructs surrogate posterior samples rather than relying on logits directly, enabling integration with other post-hoc methods (TULiP+GEN) and theoretical application to non-classification tasks (though this is untested in practice).

- **Architecture-agnostic testing.** Fig. 3 demonstrates effectiveness across MobileNet-V3, VGG-16, and RegNet-Y beyond ResNets.

## Weaknesses

### Major:

- **Fundamental disconnect between theoretical framework and practical implementation.** This is the paper's central issue. The entire theoretical edifice (Sec. 3: Thm. 3.1, Lem. 3.2, Prop. 3.3) is derived under the lazy/NTK regime with constant kernels and gradient flow. However, the practical method operates on finite-width ResNets trained with SGD-momentum, where the paper itself acknowledges "significant changes in the empirical NTK throughout the training process" (Sec. 4.1). Between the theory and the algorithm, the following heuristic substitutions are made: (i) θ_{t_s} is replaced by 0 (initialization mean), discarding the actual training trajectory; (ii) the constant K in Prop. 3.3 is replaced by a tunable hyperparameter λ; (iii) ℰ_x[Θ(x,x)] is dropped entirely; (iv) a "highly heuristic" layer-wise scaling Γ (admitted in Sec. 4.1) is introduced; and (v) max(S,0) clamping silently discards cases where the "bound" becomes negative. After these substitutions, the implemented score S in Algo. 1 is no longer a bound on ‖f_T(z) − f̂_T(z)‖ in any formal sense — it is a functional of two estimated norms and a tuned parameter. The paper presents this as "a theoretically-driven post-hoc uncertainty estimator" (Abstract) and "theoretically justified" (Intro), which overstates the connection. The theory motivates the method, but does not support it in the regime where it is applied.

- **No validation that TULiP scores track epistemic uncertainty on realistic models.** The paper's framing is that TULiP estimates epistemic uncertainty (Eq. 1, variance over training ensembles). Yet the experimental section only evaluates OOD detection AUROC/FPR95 — never whether TULiP's scores correlate with ensemble variance or Bayesian uncertainty on real networks. The synthetic validation (Fig. 2) only confirms the bound under exact lazy dynamics, not the practical pipeline. Without a direct comparison — e.g., correlating TULiP's variance with that of a deep ensemble on a realistic model like ResNet-18 on CIFAR-10 — the claim that TULiP measures epistemic uncertainty (vs. being an effective OOD heuristic informed by NTK-theoretic insights) is unsupported by evidence.

- **The layer-wise scaling Γ_l = 1/√|θ_l| is load-bearing yet poorly justified.** The paper states "our method failed to achieve consistent performance across various datasets without layer-wise scaling" (Sec. 5.2), making this step essential rather than incidental. Yet its justification rests on a single diagnostic plot (Fig. 1a–c) for one ResNet-18 on CIFAR-10, showing that Jacobian magnitude ratios at early vs. late epochs roughly scale with |θ_l|^{-1/2}. No formal argument, no test across architectures, and no comparison to alternative scaling schemes are provided. Since Γ is what bridges the gap between the NTK formalism and empirical networks, this is a methodological gap that undermines the claimed principled connection.

### Minor:

- **Near-OOD vs. far-OOD tradeoff limits practical applicability.** Fig. 4 shows that ε and λ tuned for near-OOD hurt far-OOD performance and vice versa. On ImageNet-1K, TULiP achieves FPR@95 of 64.96/48.01 (near/far) vs. ASH's 63.32/19.49 — a substantial gap on far-OOD. While consistent with the theoretical expectation from Eq. 8, this means hyperparameter selection is problem-dependent, reducing the "plug-and-play" convenience relative to simpler methods like EBO or MLS.

- **Closeness assumption (Eq. 8) is only marginally validated.** This assumption underpins Prop. 3.3 and hence the final score. Fig. 1d checks it on 256 ID and 128 per-OOD samples for a single ResNet-50 on ImageNet-1K, showing it holds "by a large margin" on average. The assumption is not formally characterized (when does it fail?), and no per-sample analysis or counterexample study is provided.

- **The non-classification generality claim is untested.** The abstract and Sec. 4.3 note that TULiP "is not limited to classification problems," but every experiment is image classification. Without even one regression or segmentation experiment, this claim remains speculative.

## Nice-to-Haves

- Ablation against naive weight perturbation (Gaussian noise on weights → prediction variance as OOD score) to isolate TULiP's theoretical machinery contributions.
- Direct correlation analysis between TULiP's surrogate variance and deep ensemble variance on at least one realistic setting.
- Evaluation on transformer architectures (noted as future work in the paper), which deviate further from the lazy regime and would stress-test the assumptions.
- More systematic computational cost analysis (wall-clock time and memory per method).

## Removed Points

- **"NTK divergence under cross-entropy invalidates A4."** The human finder flagged that recent work shows NTK diverges under cross-entropy. While theoretically relevant, A4 assumes near-perfect convergence of the *function* outputs on training data — which is empirically observed for overparameterized networks trained to low loss, regardless of whether parameters diverge. The paper explicitly motivates A4 via "overparameterized NNs trained via SGD is able to achieve near-zero training loss." This is distinct from parameter convergence. The tension is real but not fatal to A4 as stated.

- **"Computational overhead: TULiP requires 10× more forward passes than single-pass methods."** This is factually correct but is a standard tradeoff for ensemble/uncertainty methods. MCD (marked with † in Table 1) also requires multiple passes. TULiP's overhead is clearly specified (M=10), not hidden, and the paper notes TULiP is ~3× faster than ViM. This is not a unique weakness.

- **"ASh fails on different ImageNet-1K weights."** The paper itself makes this claim in the main text. I'm removing the reviewer's skepticism about this — if the paper reports it, we treat it as stated.

- **"Omitting ℰ_x[Θ(x,x)] changes the ranking of test points."** While this could be true in principle, the paper states this term is "intractable and irrelevant to z" (Sec. 4.3). Since it's a z-independent constant, its omission cannot affect ranking for OOD detection purposes (one compares scores across z). This is a reasonable design choice, not a flaw.

## Novel Insights

The most insightful observation across the reviews is that TULiP occupies an unusual position: it has genuine theoretical content (the NTK-distance bound of Thm. 3.1 is nontrivial and correctly derived) but the practical algorithm is better understood as an NTK-inspired OOD heuristic rather than a theoretically guaranteed bound. The weight perturbation mechanism essentially probes local sensitivity of the network via its Jacobian structure — which is known to correlate with OOD-ness — and the variance-matching step (γ-scaling) modulates this signal based on a comparison between trace-of-NTK and Jacobian-parameter projections. This is a reasonable and empirically effective design, but it is not formally executing what Thm. 3.1 claims. The paper would be stronger if it were more transparent about this distinction, rather than presenting the theory and practice as a seamless whole.

## Suggestions

- **Reframe the claims** to distinguish more clearly between the theoretical contribution (a bound on training fluctuations under lazy dynamics that motivates an OOD-relevant quantity) and the practical method (an NTK-inspired weight perturbation heuristic that achieves strong empirical results). Currently, the presentation implies the implementation is a direct computation of the bound.
- **Add a direct ensemble validation** on at least one realistic setting (e.g., ResNet-18 CIFAR-10): compute a 10-model deep ensemble's predictive variance, and show TULiP's surrogate variance correlates with it for both ID and OOD inputs.
- **Add a component ablation** systematically removing: (a) the D-term from Lem. 3.2, (b) the γ variance-matching, (c) the layer-wise scaling Γ, and (d) test naive Gaussian weight perturbation. This would reveal which "theoretical" components actually drive performance and which are heuristic scaffolding.
- **Provide per-sample analysis of the closeness assumption** (Eq. 8) — when does it fail, and what happens to TULiP's score on those inputs?

## Score and Decision

**Calibration:** I compared against papers with similar profiles. The NTK-theory-to-practice gap paper (GqI4fTVUXC) scored 5-6 and was rejected — its contributions were empirical and cautionary but limited. The lazy-regime training paper (XgAKt7rbXk) scored 3-5 and was rejected due to severe limitations in assumptions and experiments. The ifBDL paper with shaky NTK approximations (ohHtdp3jDi) scored 3-5 and was rejected. The OOD detection paper with practical strong results (SCALE, RDSTjtnqCg) scored 5-8 and was accepted (poster). On the high end, faithful NTK application papers like the NTK surrogate models paper (yKksu38BpM) scored 5-8 and was a spotlight.

TULiP is stronger than pure NTK-theory papers with broken assumptions because its empirical results are genuinely competitive and the theoretical framework, while not supporting the implementation as tightly as claimed, provides meaningful motivation and intuition. It is weaker than SCALE or other OOD detection papers with clean empirical contributions because its theoretical claims are substantially overstated relative to what is actually delivered. The overclaim is not cosmetic — it pervades the abstract, introduction, and framing of the method.

This paper's contribution profile is: novel and interesting theory (within its regime), strong empirical near-OOD results, but a substantial theory-practice gap that the text does not adequately acknowledge. Given that the practical method works well empirically but the primary selling point (theoretical grounding) is significantly weaker than presented, I place this below papers like SCALE (accepted poster, avg ~6.3) but above purely theoretical NTK papers with broken assumptions (rejected, avg ~4). The empirical contribution is real but the overclaim is serious.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>