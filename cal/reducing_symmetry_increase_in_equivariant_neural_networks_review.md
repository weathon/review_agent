=== CALIBRATION EXAMPLE 45 ===

# Final Consolidated Review
## Summary

This paper provides a rigorous mathematical framework for understanding and controlling "symmetry increase" in equivariant neural networks—where output representations become more symmetric than inputs, causing expressivity loss. The authors prove that for any feature space and input symmetry group, the increased symmetry admits a unique infimum determined by the feature space's algebraic structure (Theorem 3.1), develop computable algorithms to derive this infimum, and show that under standard regularity assumptions (manifold hypothesis, C^∞ approximation capability), most equivariant maps achieve this infimum (Theorem 5.2). Experiments on synthetic k-fold structures and QM9 molecular property prediction validate the theoretical predictions.

## Strengths

- **The symmetry infimum concept (Thm 3.1) provides a precise, well-defined lower bound on symmetry increase**, moving beyond qualitative observations (Curie's Principle, orbit-type discussions) to a rigorous, computable quantity. The uniqueness proof—establishing that the minimal orbit type in a fixed-point subspace is unique up to conjugation—is the theoretical linchpin that makes the entire framework well-defined.

- **The genericity result (Thm 5.2) meaningfully connects abstract representation theory to practical learning.** By showing that almost isovariant maps (those preserving symmetry up to the infimum, almost everywhere) are generic (dense) in the space of smooth equivariant maps, the paper provides a strong guarantee: for expressive enough architectures, the symmetry infimum is not just a lower bound but the typical behavior. This is a genuine advance over prior work that only identified the phenomenon without quantifying its typicality.

- **The complete orbit-type and symmetry-infimum tables for all closed subgroups of SO(3)/O(3) (Appendix E)** constitute a substantial practical contribution. These tables allow practitioners to look up the predicted degeneration behavior for any input symmetry at any feature degree, directly enabling the design guidelines from §4.2.

- **The taxonomy of degeneration types (full/axial-continuous/half-discrete) extends the prior "collapse-to-zero" framework** of Cen et al. (2024), which only addressed the most extreme case. The half-degeneration and axial-degeneration categories capture real expressivity losses that prior theory could not predict, as confirmed by the visualization experiments (Figs. 3–4).

## Weaknesses

### Major:

- **The QM9 experiment does not adequately control for confounding factors when attributing performance differences to symmetry increase.** When the paper claims that "for non-trivial feature components where molecular symmetry increase to O(3), the prediction loss is substantially higher" (§6.3), it does not isolate whether this degradation stems from symmetry increase specifically or from the fact that different-degree features have different dimensionalities and representational capacities. For instance, l=1 features (3-dimensional) and l=2 features (5-dimensional) differ in size; if a symmetry group causes l=1 to fully degenerate while l=2 remains informative, the performance gap could be partly attributed to dimension rather than symmetry. An ablation matching feature dimensions across degrees (e.g., by subsampling) would strengthen the causal claim.

- **The relationship between the high-multiplicity theory (r > dim G) and the r = 1 case used in practice is insufficiently clarified in the main text.** Proposition 4.2 requires r > dim G for Michel's criterion to be sufficient, yet the paper states predictions are "identical for the single representation case (r = 1), see §C.4." Section C.4 explains that the r = 1 results come from the Ihrig-Golubitsky criterion (with its correction term α_G), which is a different theoretical basis. This distinction is important because there could in principle be cases where the two criteria disagree, and the main text should explicitly state (1) that the high-multiplicity results are a sufficient but not necessary condition, and (2) that for r = 1 the more general criterion must be checked separately, with an explanation of why they coincide for O(3)/SO(3) specifically.

### Minor:

- **The manifold hypothesis assumption (§5.1, §A.3) is strong and unvalidated for molecular data.** While the paper acknowledges that molecular data may have self-intersections and different dimensional structures, it assumes these can be handled by finite unions of compact smooth G-invariant submanifolds. No empirical evidence is provided that QM9 data satisfies this assumption, and the consequences of violations for the genericity guarantees are not discussed.

- **The practical workflow for applying the design guidelines (§4.2) requires knowing the input symmetry group a priori.** In practice, molecular symmetries are often unknown or mixed. The paper does not discuss how to handle datasets where different samples have different (possibly unknown) symmetry groups, which limits immediate applicability.

- **The multiplicity requirement r > max_j dim M_j from Theorem 5.2 for achieving full (non-almost) isovariance is not estimated for QM9.** It remains unclear whether standard channel counts (e.g., 16 channels used in the HEGNN experiments) satisfy this condition, leaving the practical relevance of the stronger guarantee uncertain.

- **The QM9 experiments evaluate only isotropic polarizability (α).** Since different molecular properties may have different dependencies on orientational information, testing additional properties (e.g., dipole moment, which is explicitly orientation-dependent) would better validate the generality of the guidelines.

### Trivial:

- The abstract's use of "most" (italicized) is imprecise about its topological meaning; Section 5 provides the precise formulation as genericity (residual/dense open sets), but this could be signaled earlier.

## Nice-to-Haves

- A direct comparison with alternative approaches to symmetry increase, particularly Kaba & Ravanbakhsh (2023)'s method of relaxing the equivariance constraint, to demonstrate the advantages of staying within the equivariant framework.
- A practical feature-selection code snippet or decision procedure (beyond lookup tables) that practitioners can use without deep familiarity with representation theory.
- Testing on non-molecular symmetric data such as crystal structures or general point clouds to validate claims about applicability across "scientific fields."
- Training curves comparing convergence behavior with and without guideline-compliant feature selection, to assess whether symmetry increase affects optimization dynamics beyond expressivity.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Proof of Theorem 3.1 needs to address connectedness of V[H]"** — Factually incorrect. Two distinct open dense subsets of any non-empty topological space must intersect (if U, V are open dense and U ∩ V = ∅, then U ⊆ X\V, but X\V has empty interior since V is dense, contradicting U being non-empty open). Connectedness is not required.

- **"Cross-referencing error: Counterexample D.3 referenced in Section 3 but appears in Section 5"** — The paper uses proper forward referencing: "In § 5.1, we will see that this condition is in fact not sufficient." This is standard academic practice, not an error.

- **"Mismatch between TFN (theory) and HEGNN (QM9 experiment)"** — The theory in Theorem 5.2 applies to *any* equivariant parametrization with C^∞ approximation capability, not just TFN. TFN is used as a worked example for Theorem 5.1. The paper references Cen et al. (2025) for HEGNN's universal approximation property.

- **"Notation inconsistency G_O(X) vs O_G(X)"** — Formatting nitpick; both notations appear consistently in their respective contexts.

- **"Figure/table content garbled"** — PDF extraction artifacts, not paper problems.

- **"Footnote about compact Lie groups appears late"** — It appears in footnote 2 on page 2, within the preliminaries section, which is the appropriate location.

- **"Example 2.2 generators of G_x not rigorously derived"** — The paper provides the generators as statements and defers detailed calculations to the appendix (§C.3). This is standard practice for a paper with extensive mathematical content.

- **"Theorem 5.1 proof doesn't address composition with Clebsch-Gordan tensors"** — The proof in Appendix D.2 explicitly handles this through an inductive argument (Eqs. 50–59) with a product approximation lemma that bounds the error of composing approximated functions. The treatment is careful and complete.

- **"Missing related works"** — Per hard rules, cannot confirm existence of uncited works.

- **"Reproducibility concerns about undisclosed hyperparameters"** — Per hard rules, removed; the paper provides code and detailed experimental settings in the appendix.

## Novel Insights

The paper reveals a subtle structural insight: for SO(3), *all* closed subgroups satisfy the "bottleneck condition" (§C.5), meaning that any non-trivial symmetry increase from a subgroup H must pass through a unique adjacent supergroup. This gives the symmetry infimum a particularly clean compositional structure for SO(3): the infimum of a direct sum is simply the minimum of the infima of its components. The paper notes this fails for O(3) (citing C_∞ as a counterexample), creating an asymmetry between SO(3) and O(3) that has practical implications for feature design—selecting features that preserve C_∞ orbit types requires simultaneously including both even and odd parity components, a constraint not obvious from the SO(3) analysis alone.

## Suggestions

- In the main text (around §4.1), add a brief paragraph explicitly stating that the high-multiplicity results are sufficient conditions, that the r = 1 case requires the Ihrig-Golubitsky criterion with its correction term α_G, and that for O(3)/SO(3) specifically these two criteria yield identical results. Reference §C.4 for the derivation.

- For the QM9 experiment, add a dimensionality-matched control: when comparing features of degree l_0 that undergo full degeneration versus those that don't, subsample the non-degenerate features to match the effective dimensionality of the degenerate ones (which is zero for fully degenerate components), or compare by training separate MLPs with matched parameter counts to isolate the effect of symmetry increase from representational capacity.

## Axis Evaluations

- **Novelty**: Strong. The symmetry infimum concept, its uniqueness proof, and the genericity result for almost-isovariant maps are genuine theoretical contributions that go beyond prior qualitative observations of symmetry increase.

- **Technical soundness**: Generally sound, with one notable gap: the r = 1 vs. high-multiplicity theoretical justification is inadequately explained in the main text, though the appendix covers it. The proof of Theorem 3.1 is correct (the connectedness concern raised by one reviewer is spurious).

- **Empirical support**: Moderate. The synthetic experiments (k-fold structures) cleanly validate the theoretical predictions, and the QM9 experiment demonstrates real-world relevance. However, the QM9 results lack proper ablations controlling for confounding factors, and only one molecular property is tested.

- **Significance**: High for the equivariant neural network community. The framework provides both a diagnostic tool (predicting which feature degrees fail for which input symmetries) and a design principle (selecting feature components whose symmetry infimum preserves task-relevant information). This directly addresses a known failure mode in scientific ML applications.

- **Clarity**: The mathematical exposition is precise but demands significant background in Lie group representation theory and stratification theory. The main text is well-structured, but the gap between the high-multiplicity theory and practical r = 1 settings, and the practical interpretation of the guidelines, could be clearer.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
