## Summary

OC-Flow is a training-free framework for guided flow matching that recasts the generation process as an optimal control (OC) problem with an explicit running cost. Building on Pontryagin's Maximum Principle and the Extended Method of Successive Approximations (E-MSA), it derives principled update rules with monotone-improvement guarantees for both Euclidean and SO(3) manifolds, implements these efficiently via the adjoint/VJP method and an asynchronous update scheme, and shows that D-Flow and FlowGrad are special cases of the Euclidean formulation. Experiments cover text-guided image manipulation, QM9 conditional molecule generation, and all-atom peptide backbone design.

---

## Strengths

- **Theoretically grounded unification.** The optimal control framing is the first to provide a formal running-cost interpretation (Proposition 1 linking it to KL divergence), explain why FlowGrad's implicit γ→∞ limit slows convergence, and show D-Flow's single-control / L-BFGS choice as a special asynchronous case. This unified view is concrete and informative, not just analogical.

- **Novel SO(3) extension with convergence analysis.** Extending E-MSA to the SO(3) manifold, including the derivation of the co-state flow in the Lie algebra so(3), an explicit basis-decomposition implementation (Eq. 22), and a formal convergence argument (Theorem 5 + Proposition 4), is genuinely novel. No prior guided flow-matching work operates on SO(3) with theoretical support, and the peptide-design application directly motivates it.

- **Concrete memory/runtime reduction.** Replacing vanilla autograd through the ODE with the double-backwards VJP trick reduces memory from O(ND²) to O(D²), a meaningful practical advance documented in Table 1 and confirmed by the 216 s vs. 15 min comparison with D-Flow.

- **Interpretable regularization via running cost.** Proposition 1 establishes that the quadratic running cost upper-bounds the KL divergence between the guided and prior joint distributions under affine Gaussian paths, giving the hyperparameter γ a precise statistical meaning rather than being a bare tuning knob.

- **Comprehensive multi-domain evaluation.** Testing on images (CelebA-HQ with RF), small molecules (QM9 / EquiFM), and peptide backbones (PepFlow) in a single paper, with each domain probing distinct aspects of the framework, is unusually broad for this line of work.

---

## Weaknesses

- **Convergence claim is overstated.** Theorem 2 and Theorem 5 together establish only monotone non-decrease of J(θ^k) and that ε_k → 0. The paper then asserts "this establishes the convergence of the OC-Flow algorithm on the SO(3) manifold." Monotone non-decrease plus ε_k → 0 does not imply convergence to a stationary point or global optimum without additionally showing J is bounded above and that the fixed-point condition (ε_k = 0) implies the iterate is optimal. Proposition 4 states "when ε_k = 0, we have θ = θ∗ := argmax J(θ)," but this fixed-point characterization is not sufficient on its own to conclude the iterates converge to θ∗ — it only says what happens *if* convergence occurs. This is a meaningful gap, especially since the convergence guarantee is advertised as a primary contribution. The appendix may contain more detail, but the main body as written is overreaching.

- **Key undefined quantities impede reproducibility.** Three quantities appear in the main-text theorems without definition: (a) ε_γ^k in Eq. 8 — the theorem's guarantee (J increase ≥ (1 − 2C/γ)ε_γ^k) is uninterpretable without knowing what ε_γ^k is; (b) the constant C in "γ > 2C" — a practitioner is asked to satisfy a constraint involving an unknown constant; (c) "square-shaped data x" in Proposition 1 — used to motivate the image-task terminal constraint but never formally defined in the main text.

- **Table 2 caption contains a factual error.** The caption reads: "Lower LPIPS and ID indicate better performance, while higher ID and CLIP values are preferred." ID (identity preservation) appears in both halves, once as lower-is-better and once as higher-is-better. The table itself correctly shows ID ↑, so the first occurrence in the caption is wrong. This creates genuine ambiguity when reading the table.

- **QM9 Δε regression is unacknowledged.** Table 3 shows OC-Flow achieves MAE 367 on Δε (orbital gap) versus D-Flow's 355 — a 3.4% regression on the property that receives the most attention in drug design. The paper claims OC-Flow "consistently outperforms" both baselines, which is incorrect for this property. No discussion of why this regression occurs is provided.

- **Computational budget asymmetry in QM9 comparison.** The paper uses 5 L-BFGS steps (with line search) for D-Flow and 15 gradient steps for OC-Flow. L-BFGS with line search can require multiple function evaluations per step, but the exact total forward/backward pass count is not reported. Without equalizing total ODE evaluations or reporting wall-clock time, the comparison may not reflect equal compute budgets, and any residual advantage for OC-Flow is ambiguous.

- **RMSD degradation in peptide design is under-addressed.** OC-Flow(trans+rot) achieves RMSD 2.127 Å versus PepFlow's 1.645 Å — a 29% increase in structural deviation from native. The paper states OC-Flow "captures more natural structural configurations," but higher RMSD is in direct tension with this. The trade-off between energy optimization and structural fidelity deserves explicit analysis rather than being left for the reader to reconcile.

- **Differentiability requirement is an unacknowledged limitation.** OC-Flow requires ∇_{x_1} R(x_1) to exist. For many scientific reward functions (e.g., docking scores, physics-based force fields without automatic differentiation), this is unavailable. All experiments use differentiable proxies (CLIP, classifier, MadraX). This practical restriction should be stated explicitly in the limitations section.

---

## Nice-to-Haves

- **Explicit running-cost ablation.** An ablation comparing OC-Flow vs. OC-Flow-without-running-cost (γ = 0, i.e., FlowGrad) on the same task and compute budget would directly validate the benefit of the optimal control formulation beyond the unified-view argument.

- **Hyperparameter guidance.** A Pareto-frontier plot of reward (CLIP/MAE) vs. prior fidelity (LPIPS/stability) as a function of γ, or a principled heuristic for setting γ, would substantially aid practitioners. The current ablation (Table 4) over only two values of γ (0.01, 10) is insufficient.

- **Asynchronous vs. synchronous ablation.** The paper introduces the asynchronous scheme as an efficiency contribution and uses it for peptide design, but never directly compares it to the synchronous variant in terms of convergence speed and final reward quality.

- **FID/KID for image generation.** The image experiment reports only LPIPS/ID/CLIP; standard distributional quality metrics would let readers assess whether OC-Flow maintains overall generation quality beyond identity and CLIP alignment.

- **Wall-clock vs. performance trade-off curves.** A plot of reward-vs-iterations or reward-vs-compute-time against D-Flow and FlowGrad would clarify when the extra cost of OC-Flow is justified.

- **Comparison to CFG for context.** Though CFG requires conditional training (a different setting), a brief empirical calibration against it would help practitioners understand the performance gap they are accepting by using a training-free method.

- **Scaling to larger proteins.** The paper notes "protein motif scaffolding" as a future direction; even a brief feasibility demonstration on slightly larger systems would strengthen the scalability claim.

---

## Removed Points

*These points were raised by reviewers but are flagged for removal or heavy weakening upon cross-checking with the paper.*

- **"FlowGrad limiting case is not a valid reduction"** (Harsh Critic, Section 3.3): The paper explicitly derives the connection in Eq. 11, stating the joint limit γ→∞ and dt→0. The reduction is stated in the paper with the required conditions. The concern is addressed; the reduction may be informal but is not false.

- **"No statistical significance testing"**: Single-run evaluation is the norm in guided generation benchmarks (QM9, image manipulation) and is consistent with how D-Flow and FlowGrad are evaluated in their original papers. Removing this requirement is appropriate per community standards.

- **"Lipschitz assumption is practically violated"**: The paper explicitly addresses this in Section 3.1: "this assumption can be relaxed to a local Lipschitz condition if we can demonstrate that x_t^θ is bounded, which can be safely assumed provided that appropriate regularization techniques are applied." The concern is noted and reasonably deflected.

- **"Unfair comparison with D-Flow (L-BFGS vs gradient)"**: L-BFGS is more expensive per step than gradient descent, so giving D-Flow its preferred optimizer is asymmetric in favor of the baseline. This is not a weakness of the paper; the concern is kept partially as a weakness only because the total step count is 15 vs. 5 (see Weaknesses above) — not because of the optimizer choice.

- **"Proposition 3 formatting inconsistency / numbering error"**: The paper refers to the proposition as "Proposition 2 provides a lower bound" in text immediately after labeling it "Proposition 3." This is a minor numbering/editorial slip, not a scientific error. Removed as pure formatting issue.

- **"No other SO(3) guided baseline"**: The paper explicitly positions this as one of the first guided flow methods on SO(3). The absence of a competitor is a limitation of the field, not of the paper. Comparing to the unconditional baseline is the appropriate comparison in this setting.

---

## Novel Insights

The most genuinely novel synthesis across the three reviews is the connection between the running cost's role as KL regularizer and the empirically observed trade-off curves: by treating γ as a formal control on KL divergence (Proposition 1) rather than as an ad-hoc hyperparameter, OC-Flow offers a principled explanation for why FlowGrad (implicit γ→∞) tends to produce distorted outputs and why D-Flow (single control, L-BFGS learning rate ≈ variable γ) offers implicit regularization through projection. This reframing transforms what has been a poorly understood engineering trade-off in backprop-through-ODE guidance into a mathematically characterized optimization problem — which is the paper's most durable contribution regardless of the empirical results.

---

## Evaluation

- **Novelty:** High. The OC framing with a running cost that admits a KL interpretation, and especially the SO(3) extension, represent genuine advances over FlowGrad and D-Flow. The unification argument is concrete and provable, not merely analogical.
- **Technical soundness:** Moderate-to-high. The core algorithmic derivations (E-MSA adaptation, VJP trick, SO(3) co-state flow) are technically careful. However, the convergence claims in both the Euclidean and SO(3) settings are stated more strongly in the main body than the proofs actually support.
- **Empirical support:** Moderate. The three-domain evaluation is commendable. However, an unacknowledged regression (QM9 Δε), a significant RMSD degradation in the most novel experiment, and an unequalized compute budget in the molecule task leave empirical support incomplete.
- **Significance:** High. Principled training-free guidance on SO(3) flow models has immediate value for protein and peptide design, an area of active and high-stakes scientific interest.
- **Clarity:** Moderate. The paper is well-organized overall, but key quantities (ε_γ^k, constant C, "square-shaped data") referenced in the main theorems are not defined there, and the Table 2 caption error introduces genuine confusion.

---

## Suggestions

1. **Tighten the convergence claim.** Either strengthen Proposition 4 to include a convergence-in-iterate result (e.g., using compactness of the control set), or revise the text to accurately say the algorithm monotonically improves J(θ^k) and that any fixed point satisfies the optimality condition — without claiming full convergence to global optimum from the current proof alone.
2. **Define ε_γ^k and constant C in the main body.** Even one sentence explaining what ε_γ^k represents and how C depends on L (with a rough order-of-magnitude) would make Theorem 2 interpretable without appendix access.
3. **Define "square-shaped data" formally.** This term is used in a proposition that motivates the design choice in the image experiment; it must have a precise definition accessible to readers.
4. **Correct Table 2 caption.** Remove the first occurrence of "ID" from the "lower is better" list; the ↑ arrow in the table is correct.
5. **Address the QM9 Δε regression explicitly.** One or two sentences analyzing why D-Flow outperforms on orbital gap despite underperforming elsewhere would strengthen scientific credibility.
6. **Equalize compute budget in QM9.** Report wall-clock time for OC-Flow (15 steps) vs. D-Flow (5 L-BFGS steps) to allow readers to assess whether the comparison is fair and whether OC-Flow's advantage is compute-adjusted.
7. **Address the RMSD trade-off in peptides.** Explicitly acknowledge that energy optimization comes at the cost of RMSD fidelity; a Pareto curve or a sentence acknowledging the trade-off would be more honest than claiming better structural naturalness.
8. **Add differentiability limitation to Section 6.** Explicitly note that OC-Flow requires differentiable reward functions (or differentiable surrogates), as this is a real constraint for deployment in simulation-heavy scientific pipelines.

MY FINAL SCORE: <pineapple>6.8</pineapple>