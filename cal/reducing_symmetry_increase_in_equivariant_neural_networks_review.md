=== CALIBRATION EXAMPLE 26 ===

# Final Consolidated Review
## Summary
This paper studies **symmetry increase** in equivariant neural networks: for symmetric inputs, equivariant maps can force the output to have strictly larger symmetry than the input, causing representational collapse. The paper’s core contribution is a rigorous characterization of this phenomenon via a **symmetry infimum** determined by the feature space, together with computable orbit-type algorithms, design guidelines for choosing feature spaces, and genericity results showing that under regularity assumptions most sufficiently expressive equivariant maps attain this predicted infimum.

## Strengths
- **A genuinely substantive theoretical advance over prior partial accounts of symmetry collapse.** The paper does more than restate Curie’s principle. It introduces the symmetry infimum \(I_G(Y,H)\) as a feature-space-determined lower bound on output symmetry increase and proves its uniqueness (Thm. 3.1), then connects existence of symmetry-preserving maps to orbit-type conditions (Thm. 3.2, 3.3). This gives a clean and reusable mathematical object for reasoning about when equivariant features can or cannot preserve task-relevant symmetries.
- **The treatment of nontrivial kernels is careful and practically relevant.** Section 3.2 does not ignore the common case where the output action is unfaithful; instead it defines the projection operator \(p_Y\) and reframes the goal as relative isovariance. This is important for realistic ENN outputs where some symmetry increase is designed in rather than accidental.
- **The paper turns abstract orbit-type theory into concrete computable tools.** Algorithm 1 and Algorithm 2, together with the sufficiency result for high-multiplicity representations (Prop. 4.2), provide an operational procedure for determining orbit types and symmetry infima. The extensive tables for \(SO(3)\)/\(O(3)\) subgroups and irreps are likely to be a useful reference for future work.
- **The degeneration taxonomy is sharper than the usual “collapse-to-zero” story.** The distinction between full, axial, and half degeneration is not cosmetic: it clarifies that failure modes are not binary, and Example 4.3 plus the tables show how specific feature degrees induce distinct classes of loss of orientation information.
- **The genericity result is ambitious and nontrivial.** Section 5 does not merely provide existence; it argues that, under manifold and approximation assumptions, almost-isovariance relative to the predicted infimum is generic (Thm. 5.2), and even exact relative isovariance can be obtained with sufficient multiplicity. This is a substantial theoretical statement, not a minor extension.
- **The synthetic experiments do directly validate the representation-theoretic predictions.** The visualization and symmetric-graph experiments are aligned with the theory: they test whether the predicted collapses actually appear in TFN/HEGNN embeddings for \(k\)-fold structures, and the reported binary separation in §6.2 is consistent with the specific orbit-type predictions from Example 4.3/Table 1.
- **The QM9 analysis, while limited, is more than a generic benchmark add-on.** The experiments intentionally relate per-degree features to predicted degeneracies for different molecular point groups, and the case studies in §F.3.2 show examples where fully degenerate components correlate with worse prediction behavior.

## Weaknesses
### Major:
- **The empirical section does not adequately validate the paper’s stronger practical claim that the framework yields an effective method for *reducing* harmful symmetry increase through architecture/feature design.**  
  This criticism is supported by the actual paper. The strongest claims in the abstract/introduction are practical: the paper says it provides a “principled framework for its reduction,” “practical guidelines for feature design,” and that these guidelines “effectively reduce symmetry increase.” However, the experiments mostly do one of three things:  
  1. visualize collapse in randomly initialized models (§6.1),  
  2. measure indistinguishability in randomly initialized encoders (§6.2), or  
  3. freeze a pretrained encoder and mask degrees post hoc on QM9 (§6.3).  
  None of these is a controlled intervention where one **constructs** feature spaces according to the proposed guidelines and shows improved performance relative to a standard design on a task where preserving orientation/symmetry information is essential. As a result, the paper strongly validates the *predictive* theory of symmetry increase, but only weakly validates the *prescriptive* claim that the framework improves practical architecture design.
- **The practical significance is somewhat overstated because the harmfulness of symmetry increase is task-dependent, and the paper’s empirical task alignment is incomplete.**  
  The introduction frames symmetry increase as a “critical vulnerability” causing expressivity degradation, but the paper itself later distinguishes “orientation-dependent tasks” from “general tasks” (§4.2). For invariant scalar targets, symmetry increase is not uniformly harmful; indeed §4.2 acknowledges that one should mainly avoid forms causing severe fixed-point compression or annihilation. The current empirical validation does not fully resolve this mismatch because the only real-world task is **isotropic polarizability prediction** on QM9 (§6.3), which is not an orientation-sensitive target. This does not invalidate the theory, but it does weaken the breadth of the motivational framing and the practical ML claim.
- **There is a real theory-to-practice gap between the generic map results in §5 and the behavior of trained finite networks.**  
  Theorem 5.2 is mathematically interesting, but it is stated for parametrizations with \(C^\infty\) approximation capability and concludes that almost-isovariance is generic on unions of smooth compact \(G\)-submanifolds. The paper does not substantially discuss whether gradient-based training in finite-width ENNs tends to realize these generic maps, or whether optimization biases could favor nongeneric solutions where symmetry increase exceeds the infimum. This is especially relevant because the empirical sections do not directly test trained models on symmetry-sensitive tasks. Thus, while the theorem is technically sound within its assumptions, its implications for practical training are not yet convincingly established.
- **The computational/practical story is strongest for high-multiplicity representations, while many realistic ENN settings operate in lower-multiplicity regimes.**  
  Proposition 4.2 gives a convenient sufficiency criterion only for “high-multiplicity” representations where each nonzero isotypic multiplicity exceeds \(\dim G\). The paper does note that for some cases predictions agree with \(r=1\) and provides tables in the appendix, but the main computational framework and guarantees are most straightforward under the high-multiplicity assumption. Since practical ENNs often use limited channels, this reduces the immediate accessibility of the method as a design tool for typical architectures.

### Minor
- **The paper is mathematically dense to the point that many ML readers will struggle to extract the actionable design rule.**  
  The core message is there, especially in §4.2, but the path from orbit types / fixed-point spaces / stratification to a simple practitioner workflow is not distilled enough in the main paper. A compact “how to use this for architecture design” recipe would improve accessibility substantially.
- **The QM9 experiment is suggestive rather than causal.**  
  The masked-degree setup shows correlations between theoretically degenerate components and error, but because it uses a frozen pretrained encoder plus different masking strategies, it does not completely isolate whether the observed degradation is due specifically to symmetry increase rather than broader representation-quality effects induced by the masking protocol.
- **The paper’s scope is narrower than some of its framing suggests.**  
  Although Example 2.1 involves \(H \times S_n\), the concrete computational development, tables, and most analysis are focused on \(SO(3)\)/\(O(3)\) representation spaces. This is fine, but the paper would benefit from more explicitly emphasizing that its most mature computational guidance is currently in this setting.

### Trivial
- **A brief summary table or checklist in the main text would help readers map task type (orientation-dependent vs. general) to recommended feature-space choices.**  
  This is not a flaw in technical content, but a missed opportunity to make the contribution easier to use.

## Nice-to-Haves
- Compare a baseline ENN architecture against a version explicitly constructed using the symmetry-infimum guidelines on at least one **symmetry-sensitive** downstream task.
- Add an experiment with **trained** models where preserving orientation information is necessary, to test whether the genericity results in §5 are reflected in optimization outcomes.
- Provide a practitioner-oriented workflow: given input symmetry \(H\), target type, and candidate feature degrees, how should one choose \(Y\)?
- Clarify more explicitly how to use the framework in low-multiplicity/channel regimes, even if only via heuristics or caveats informed by the appendix tables.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Only TFN/HEGNN are evaluated, so applicability to EGNN/SE(3)-Transformer/etc. is unclear.”**  
  Removed because this is mostly a generic request for more baselines rather than a substantive flaw here. The paper’s theoretical claims are not architecture-specific in the same way, and the current experiments already use two distinct equivariant backbones.
- **“The paper should show state-of-the-art performance improvements.”**  
  Removed because the paper is fundamentally a theory/design paper, not a new predictive model claiming SOTA.
- **“Predictions for \(r=1\) are merely deferred to external tables, so the framework is not self-contained.”**  
  Weakened/removed as a major criticism. The paper does provide substantial appendix material and explicitly states where the low-multiplicity results are tabulated; this is a limitation of emphasis, not a failure of correctness.
- **Any criticism doubting cited tools, datasets, or references.**  
  Removed per instruction; the paper cites the code, QM9, PointGroup, etc., and these should be treated as real and available.

## Novel Insights
The most interesting synthesis here is that the paper’s strongest contribution is not merely identifying collapse, but **separating unavoidable, feature-space-induced symmetry increase from accidental additional increase**. The kernel-aware notion of relative isovariance and the symmetry infimum together give a principled language for saying what symmetry loss is structurally forced by the chosen representation and what symmetry loss reflects avoidable architectural mismatch. This is the conceptual bridge that makes the work more than a catalog of failure cases. At the same time, the review consensus and the paper itself make clear that this bridge is currently much stronger on the **diagnostic/predictive** side than on the **interventional/design-validation** side.

## Suggestions
- Add one end-to-end experiment where architectures are **constructed according to the proposed feature-design guidelines** and compared against standard feature choices on a task requiring orientation discrimination.
- Moderate the framing in the introduction/abstract so that claims about “critical vulnerability” and “practical reduction” better match the current evidence; emphasize that the paper most strongly establishes a predictive and diagnostic theory.
- Add a concise practitioner section or figure: input symmetry \(\rightarrow\) compute \(I_G(Y,H)\) \(\rightarrow\) identify degeneration type \(\rightarrow\) choose/avoid specific feature components depending on task type.
- Expand discussion of how §5’s genericity results should be interpreted for finite trained networks and what failure modes may remain under gradient-based optimization.
- Strengthen the low-multiplicity discussion in the main text, since this is where many practical ENNs operate.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
