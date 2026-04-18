Now I have enough context from the calibration papers. Let me write the final review.

## Summary

This paper shows that memory-augmented Transformers (Memformers), which retain intermediate attention values via memory registers across layers, can structurally implement update rules resembling Linear First-Order Methods (LFOMs), including conjugate gradient descent and momentum methods, during in-context learning on linear regression tasks. The authors provide two propositions establishing structural analogies between Memformer architectures with memory registers and LFOM/CGD update rules, and present empirical results showing that trained Memformers can achieve competitive or better performance than standard conjugate gradient on small-scale quadratic problems.

## Strengths

- **Clear and motivated architectural insight.** The idea of adding simple per-layer memory registers to accumulate past attention outputs—naturally enabling gradient momentum—makes architectural sense and is clearly connected to the LFOM framework. The update rules in Eqs. (17)–(18) and (19)–(20) are cleanly presented and easy to follow.

- **Empirical signal that memory helps in-context optimization.** The experiments do demonstrate that adding memory improves performance over standard linear transformers on the chosen tasks, and that multi-head attention further improves results (Figure 5). The finding that meta-learned shared parameters can sometimes match per-instance CGD (Figures 1b, 2a) is notable, even if the comparison has caveats.

- **Honest scope discussion.** The paper explicitly states (Section 6) that it is "not advocating for Transformers as replacements for established optimization methods in practical applications" and acknowledges that Memformers do not radically outperform preconditioned GD on quadratics.

## Weaknesses

### Fatal
None.

### Major

- **Theoretical claims of "implementing CGD/LFOM" are overstated relative to what is proved.** Propositions 1 and 2 establish structural analogies between the Memformer's update equations and LFOM/CGD recurrences, but they do not prove that the architecture can exactly reproduce CGD or general LFOM iterations on arbitrary inputs. For CGD specifically, the step sizes α_n and conjugacy coefficients γ_n are data-dependent quantities computed from current and previous gradients at each iteration (as defined in the paper's own Section 2.2). The Memformer architecture (Eqs. 17–18) uses fixed learned scalars α_ℓ and γ_ℓ per layer, which cannot adapt to individual data instances. The proof sketches in the body only describe qualitative correspondences ("mimics," "matches CGD applied to the loss") without specifying exact parameter settings or proving equivalence on all admissible data. The paper's own Section 3.3 acknowledges that "CGD-like" means only that learned parameters "may not match the exact CGD parameters for individual observations," making the stronger Proposition 1 claim misleading. This echoes concerns raised in reviews of similar papers (ZIFkrT1GwM, uqLQjtSdFN) where expressivity constructions were noted as insufficient without learning guarantees or verification that such constructions arise from training.

- **Gap between "can represent" and "learns."** The paper defines "learning an algorithm" to include both (1) existence of parameter settings for implementation and (2) achieving competitive performance after training. Criterion (1) is an expressivity claim, not established rigorously as discussed above. Criterion (2) is supported only by loss curves, with no mechanistic analysis of what the trained model actually computes. There is no inspection of learned parameters (α_ℓ, γ_ℓ, Γ_ℓ, A_ℓ) to determine whether they correspond to known LFOM algorithms, no probing of memory register contents, and no ablation isolating the memory mechanism. Reviews of analogous work (uqLQjtSdFN, Reviewer 5) flagged this same pattern: "without theoretical or empirical evidence showing that such behavior indeed manifests in trained transformers... the paper cannot serve as a convincing contribution to our understanding of ICL, rather only confirming the expressivity of the transformer architecture."

- **Unfair comparison framing with CGD.** The "surprising" claim that Memformers outperform CGD (Main Contributions (2)) requires careful interpretation. In Figure 1a (without preconditioning), the Memformer clearly underperforms standard CGD. The superiority only appears when the Memformer is augmented with trainable preconditioners (A_ℓ, B_ℓ) that effectively transform it into a quasi-Newton-like method, while the baseline CGD remains a first-order method without preconditioning. Comparing a meta-learned preconditioned method against a non-preconditioned, distribution-agnostic competitor is not a fair contest of algorithmic capability. The paper would benefit from comparing against a preconditioned CGD variant or at least acknowledging this asymmetry more prominently.

### Minor

- **Narrow experimental scope.** All experiments use d=5, n=20, and at most 4 layers. There is no scaling study in dimension or context length, no systematic variation of condition number beyond one isotropic/non-isotropic pair, and no evaluation on non-quadratic tasks. This limits confidence that the findings generalize beyond the toy regime.

- **Figure 4 evaluates on training data.** The caption and Section 4 state results on "small batch training data." A Memformer tested on the same distribution it was trained on—particularly with very small batches (B=1)—does not demonstrate generalization. This result should be re-evaluated on held-out test data.

- **Multi-head attention analysis is heuristic.** Section 5 claims multi-head attention improves performance through "diverse preconditioning" and "ensemble-like behavior," but provides only a 1-head vs. 5-head comparison with no controlled ablation, parameter count matching, or formal justification.

### Trivial
None.

## Nice-to-Haves

- Analyzing the learned parameters (α_ℓ, γ_ℓ, Γ_ℓ) after training to verify whether they converge to CGD-like or LFOM-like values, or plotting optimization trajectories against known algorithms in parameter space.
- Comparing against a directly meta-learned simple parametric optimizer (e.g., learned step-size GD or momentum) with similar parameter budget, to isolate whether the Memformer architecture itself provides benefits beyond generic meta-learning.
- Extending experiments to higher dimensions and non-quadratic losses to assess generalization of the findings.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh critic's claim that the architecture lacks "degrees of freedom" to implement CGD on arbitrary quadratics.** While CGD's adaptive α_n and γ_n are indeed data-dependent, the expressivity question is about whether there exist parameterizations that could reproduce CGD for specific distributions—this is a more nuanced point than simply declaring it impossible.

- **Demands for convergence guarantees, loss landscape analysis, or theoretical proofs that training recovers LFOM parameters.** While these would strengthen the paper, they are not standard in this particular literature (cf. Ahn et al. 2024, which the current paper extends, also does not provide such analysis). Demanding this falls under "nice-to-have" rather than a core flaw.

- **Demands for experiments on non-quadratic objectives or real-world data.** The paper explicitly scopes to linear regression (Section 2) and acknowledges this limitation (Section 6). This is scope creep.

- **Criticism that the paper should compare against preconditioned CGD.** This is a valid minor concern for fairness of framing, but is not a standard requirement. The paper's primary contribution is showing the Memformer architecture's capability, not benchmarking against the best possible optimizer.

- **Demands for computational cost comparison between Memformer forward passes and CGD.** The paper explicitly disclaims practical application as a goal (Section 6).

- **Formatting and notation nits** (e.g., Γ dimensions mismatch, proof details in appendix). These are acknowledged in the paper and do not affect the central claims.

## Novel Insights

The central finding—that adding simple per-layer memory registers to linear transformers enables them to accumulate past gradients and structurally implement momentum-like update rules—is architecturally natural and aligns with intuition from optimization theory. However, the gap between this structural observation and the stronger claim that Memformers "learn" or "implement" specific algorithms like CGD remains unbridged. The most valuable insight in the paper is actually the empirical observation (Figure 2b) that memory provides no benefit on isotropic data, confirming that momentum-type methods derive their advantage specifically from non-isotropic curvature—a connection that is well-understood in optimization but had not been demonstrated in the transformer ICL setting.

## Suggestions

- Tone down Propositions 1 and 2 to reflect that they establish structural analogies (the Memformer recurrence has the same functional form as LFOM/CGD-like updates with fixed coefficients), rather than claiming they "implement" CGD or general LFOM iterations. Acknowledge that CGD's data-dependent coefficients cannot be captured by fixed per-layer parameters.
- Add a mechanistic analysis: after training, compare learned α_ℓ, γ_ℓ to theoretical CGD values computed on held-out problem instances. This would directly address whether the model is learning "CGD-like" behavior or something qualitatively different.
- Move Figure 4's evaluation from training data to test data, and compare the Memformer against a simple meta-learned momentum GD (with similar parameter count) to isolate the contribution of the memory architecture.

## Score and Decision

**Calibration papers compared against:**
- uqLQjtSdFN (Functional Gradients, ICL): avg score ~3.7, Reject — similar pattern of expressivity-only results without learning guarantees, overclaimed theoretical contributions.
- ZIFkrT1GwM (Pretrained Transformers are Deep Optimizers): avg score ~5.25, Reject — similar pattern of constructions not being learned, lack of practical insight.
- AC9FsaVIpk (Gating is Weighting / GLA): avg score ~5.8, Reject — similar pattern of construction showing subset rather than equivalence, gap between theory and what emerges from training.
- r3DF5sOo5B (Transformers Learn Multi-step GD with CoT): avg score ~7.3, Accept Spotlight — stronger paper with actual training dynamics analysis and convergence guarantees, which this paper lacks.

This paper sits below AC9FsaVIpk (which at least has more rigorous landscape analysis for GLA) and around uqLQjtSdFN in terms of the gap between claimed and demonstrated contributions. The structural insight about memory registers is sound, but the theoretical claims overreach what is proved, and the empirical evaluation is too narrow and lacks mechanistic analysis. The paper would need significantly more rigorous theoretical treatment and deeper empirical investigation to support its central narrative.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>