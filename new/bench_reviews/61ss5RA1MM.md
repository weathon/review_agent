Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper proposes OC-Flow, a training-free framework for guided flow matching formulated as an optimal control problem. It introduces a running cost that provably bounds KL divergence from the prior (Proposition 1), derives convergence guarantees for the E-MSA algorithm in both Euclidean space (Theorem 2) and on SO(3) (Theorem 5), shows that D-Flow and FlowGrad are special cases, and demonstrates empirical improvements on text-guided image manipulation, conditional molecule generation, and peptide design.

## Strengths

- **Optimal control perspective provides genuine conceptual clarity.** The co-state flow formulation (Section 3.1, Equation 7) derives a principled update rule with weight decay and learning rate from first principles, rather than the ad-hoc gradient recipes of prior work. The running cost as an explicit regularizer with a tunable γ parameter is a meaningful practical improvement—Table 4 confirms that increasing γ from 0.01 to 10 improves faithfulness metrics (ASP: 94.8→96.0, MSP: 64.4→69.9) at moderate cost in MAE.

- **Proposition 1 provides formal justification for the running cost.** Equations 5–6 prove that the expected running cost upper-bounds the KL divergence between the prior and guided joint distributions, and combined with terminal distance, bounds the KL between marginals. This is a theoretical contribution that prior methods like FlowGrad lack entirely (Table 1 shows "Running Cost: 0" for FlowGrad).

- **Consistent empirical improvements across three diverse domains.** On text-guided image manipulation (Table 2: LPIPS 0.207 vs 0.302 for FlowGrad), conditional molecule generation (Table 3: best MAE on 5 of 6 properties), and peptide design (Table 5: OC-Flow(trans+rot) achieves best MadraX energy, stability, affinity, and IMP), OC-Flow demonstrates broad applicability. The memory improvement from O(ND²) to O(D²) via the adjoint method with VJP (Section 3.2.1) is a genuine practical advance, and Section 6 reports image sampling in 216s vs. 15 minutes for D-Flow.

- **SO(3) theoretical development is non-trivial.** The adjoint equation with the ad* term (Algorithm 2, Step 5), the extended Hamiltonian construction (Equation 18), and the Riesz representation trick for computing the co-state (Equation 22) correctly address the geometric complications of extending optimal control to Lie groups. Theorem 5 provides convergence guarantees for SO(3), which no prior work offers.

## Weaknesses

### Fatal
None.

### Major

- **Theory-practice gap in convergence guarantees.** Theorems 2 and 5 require global Lipschitz continuity of the reward function, prior model, and their derivatives. Neural network models used in all experiments (Rectified Flow, EquiFM, PepFlow) are not globally Lipschitz. The paper acknowledges this (Section 3.1, line 94: "this assumption can be relaxed to a local Lipschitz condition if we can demonstrate that $x_t^\theta$ is bounded, which can be safely assumed provided that appropriate regularization techniques are applied"). While the relaxation argument is plausible and the paper cites work on Lipschitz continuity of deep learning models (Gouk et al., 2021; Khromov & Singh, 2024), the gap remains: boundedness of $x_t^\theta$ is itself not proven but merely assumed to follow from the regularization whose convergence the theorem is supposed to characterize. This does not invalidate the conceptual value of the theory—the convergence analysis provides insight into the role of γ and the monotonic improvement property—but it means the formal guarantees do not directly apply to the actual models used. The convergence constants C and D are functions of the unspecified Lipschitz constant L, making it impossible to assess whether the bounds are tight or vacuous.

- **SO(3) contribution is undervalidated experimentally.** Contribution 3 claims "one of the first guided flow-matching algorithm on the SO(3) manifold with theoretical grounds," yet the peptide design experiment (Table 5) compares only against the unconditional PepFlow baseline. No guided SO(3) baseline is included—even simple alternatives like applying D-Flow to SO(3) rotations alone, or naive gradient descent on rotations without the optimal control framework. The improvements are marginal on the primary metric (IMP: 14.3% → 15.0%) and come with significantly degraded structural accuracy: RMSD increases from 1.645 to 2.127 (a 29% degradation) and BSR decreases from 0.874 to 0.869. The paper claims the method "consistently outperforms the baseline" and lists BSR among improved metrics, but BSR actually decreases. Without a guided SO(3) baseline, the paper cannot establish that its SO(3) optimal control formulation provides any benefit over simpler alternatives. The "first" claim is technically true but substantively unsupported.

### Minor

- **FlowGrad "special case" characterization is more framing than deep unification.** Calling FlowGrad a special case of OC-Flow with γ→∞ (Section 3.3) means "without the running cost (OC-Flow's central mechanism), you get FlowGrad"—this is accurate but not a deep insight. The D-Flow connection (n=1 control term in the asynchronous setting) is more substantive but is an analogy rather than strict subsumption, since D-Flow uses L-BFGS while OC-Flow uses the co-state flow. Table 1 presents these as if on equal theoretical footing, but the "unified view" does not provide algorithmic insight into when each approach is preferable.

- **The ID metric in Table 2 slightly favors FlowGrad** (0.737 vs 0.732 for OC-Flow). Since identity preservation is precisely what the running cost should help with, this deserves acknowledgment. The gap is small and OC-Flow is much better on LPIPS (0.207 vs 0.302), but the paper should discuss this trade-off explicitly.

- **No variance or significance reported for peptide design metrics** (Table 5). With only marginal improvements (IMP: 14.3% → 15.0%), statistical significance cannot be assessed. This makes it difficult to determine whether the SO(3) guidance effect is real or within noise.

- **Proposition 1 assumptions are unvalidated.** The "Affine Gaussian Probability Path" and "square-shaped data with non-zero probability path" assumptions (Equations 5–6) are not validated for any experimental setting. The paper claims the square-like assumption is "satisfied" for images (Section 5.1) without justification. The constant C in Equation 5 is unspecified.

### Trivial
None.

## Nice-to-Haves

- **Guided SO(3) baseline comparison** — Even a simple gradient-descent-on-rotations baseline would substantially strengthen the SO(3) contribution validation, which is a headline claim.
- **γ→∞ ablation** — Table 4 ablates γ values but does not include the limiting case γ→∞ (equivalent to removing the running cost entirely), which would directly test whether the running cost mechanism drives improvements versus other algorithmic differences.
- **Wall-clock time comparison** across all experiments, not just the single mention in Section 6.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that the convergence circularity is "fatal"**: The paper acknowledges the Lipschitz gap and provides a reasonable (if imperfect) relaxation argument citing prior work on neural network Lipschitz continuity. This is a standard theory-practice gap in ML, not a fatal flaw. Demoted to Major.

- **Harsh critic's claim that the update rule is "just momentum SGD"**: While Equation 7 can be reparameterized as momentum SGD with weight decay, this is reductive—the value of the OC formulation lies in the principled derivation and the KL-divergence bound on the running cost, not in the specific algebraic form of the update. Removed as a weakness.

- **Harsh critic's complaint about 5 L-BFGS steps "handicapping" other methods**: The paper explicitly states this is "to be comparable to D-Flow," which is a standard fair comparison practice. Using more steps for one method would be an unfair comparison. Removed.

- **Harsh critic's concern about "large gap between guided models and the Classifier oracle"** (0.314 vs 0.046 for μ): The classifier oracle is a trained conditional model, not a guided method—it represents a fundamentally different approach with different costs. Comparing guided methods to a conditional oracle is informative but the gap is expected and not a weakness of OC-Flow specifically. Removed.

- **Strength Finder's claim that "OC-Flow matches FlowGrad's O(D²) memory while D-Flow requires O(ND²)" as a strength**: This is already captured in the paper's Table 1 and is a practical efficiency improvement, which is noted in the strengths section. Kept but not doubled.

- **Strength Finder's claim about "first guided flow matching algorithm on SO(3) with theoretical grounding" as a core strength**: This is a contribution claim, but as noted in the Major weaknesses, the experimental validation for SO(3) is thin. Kept the theoretical development as a strength but moved the "first" claim to the weakness section since it is unsupported by proper baselines.

## Novel Insights

The paper reveals a tension common in ML theory papers: the optimal control framework provides genuine conceptual clarity—the running cost's role in bounding KL divergence, the principled derivation of the weight-decay/learning-rate parameterization, the co-state flow as a generalization of the adjoint method—but the formal convergence guarantees operate under assumptions that the practical algorithms cannot satisfy. The most valuable contribution may not be the theorems themselves but the structural insight they provide: that FlowGrad's instability stems from the absence of a running cost (γ→∞), and that D-Flow's strong prior faithfulness comes from constraining the optimization to a single control variable (n=1). The SO(3) extension is theoretically sound but the peptide design experiment exposes a fundamental trade-off: optimizing for energy (MadraX) degrades structural accuracy (RMSD), suggesting that the reward-prior balance on manifolds may require more careful tuning than in Euclidean settings.

## Suggestions

- Add at least one guided SO(3) baseline to the peptide design experiment—e.g., apply gradient descent on SO(3) rotations directly using the reward gradient, without the OC framework. This would validate whether the OC formulation provides benefits beyond simple gradient-based guidance on the manifold.
- Report variance across multiple runs for Table 5 and discuss the RMSD/BSR degradation of OC-Flow(trans+rot) relative to the PepFlow baseline. The current framing of "consistently outperforms" is inaccurate given these regressions.
- Consider adding the γ→∞ ablation to Table 4, which would directly quantify the contribution of the running cost mechanism versus other algorithmic differences.

## Evaluation on Axes

**Originality**: The optimal control perspective on guided flow matching is a natural but meaningful reframing. The SO(3) extension with convergence guarantees is novel. The unification of D-Flow and FlowGrad is useful but limited in depth. Moderate originality.

**Importance of research question**: Training-free guided generation for flow matching is important and timely, especially the extension to SO(3) for scientific applications. High importance.

**Claims support**: The empirical claims are largely supported in Euclidean settings but partially undermined in the SO(3) setting by the lack of guided baselines and the unacknowledged RMSD/BSR degradation. The theoretical claims are technically correct under stated assumptions but those assumptions do not hold for the models used. Partial support.

**Soundness of experiments**: Euclidean experiments are sound with appropriate baselines. SO(3) experiment is incomplete—no guided baseline, no variance reported, and overclaimed results. Moderate soundness.

**Clarity of writing**: The paper is reasonably well-structured. The presentation of the OC framework and the distinction between D-Flow, FlowGrad, and OC-Flow in Figure 1 is clear. Some overclaiming in the experimental sections weakens clarity of contribution.

**Value to community**: The OC framework provides a useful conceptual tool for understanding and improving guided flow matching. The SO(3) theory, even if undervalidated experimentally, lays groundwork for future work. Moderate-to-high value.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| RB-Modulation | /home/wg25r/review_agent/human_reviews/bnINPG5A32.md | 8.0 | Training-free guidance via stochastic optimal control for diffusion. Much stronger empirical validation and practical impact. OC-Flow is below this. |
| FoldFlow | /home/wg25r/review_agent/human_reviews/kJFIH23hXb.md | 8.0 | Flow matching on SE(3) for proteins. Far more thorough experimental validation on SO(3)/SE(3). OC-Flow's SO(3) contribution is much thinner. |
| SMC Diffusion | /home/wg25r/review_agent/human_reviews/vi3DjUhFVm.md | 7.25 | Training-free guidance via SMC. Cleaner theoretical framework. OC-Flow has broader scope but weaker theory-practice alignment. |
| Lipschitz SGM | /home/wg25r/review_agent/human_reviews/r3cWq6KKbt.md | 6.0 | Theoretical paper on Lipschitz estimates for score models. Similar theory-practice gap, but purely theoretical. OC-Flow has more practical contributions. |
| FIG | /home/wg25r/review_agent/human_reviews/fs2Z2z3GRx.md | 6.0 | Guided flow matching for inverse problems with theoretical justification. Similar level of novelty. OC-Flow has broader scope but more overclaiming. |
| Operator Networks PDE | /home/wg25r/review_agent/human_reviews/xpmDc76RN2.md | 2.33 | Circular convergence guarantees. OC-Flow has a similar Lipschitz gap but substantial empirical contributions that paper lacked. Clearly above this. |
| Nonconvex SGD | /home/wg25r/review_agent/human_reviews/PwoplYNsBI.md | 2.5 | Assumptions break down for NNs, weak empirical support. OC-Flow is clearly above this due to real empirical results. |

OC-Flow sits between the medium-scoring anchors (FIG at 6.0, Lipschitz SGM at 6.0) and the low-scoring anchors. It has genuine contributions—the OC perspective, the KL bound, the SO(3) theory, and empirical results across 3 domains—but these are partially undermined by the theory-practice gap and the thin SO(3) experimental validation. Compared to FIG (6.0, accepted), OC-Flow has broader scope but weaker rigor and more overclaiming. I place it slightly below the acceptance borderline.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>