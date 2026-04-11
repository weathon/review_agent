=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary
This paper provides a rigorous mathematical analysis of symmetry increase in Equivariant Neural Networks (ENNs), a phenomenon where symmetric inputs are mapped to outputs with higher symmetry, degrading expressivity. The authors introduce the concept of a *symmetry infimum*—a unique lower bound on the increased symmetry determined by the feature space—and prove its existence and computability via orbit-type analysis. They develop algorithms to compute this infimum for SO(3) and O(3) representations, offer practical guidelines for feature design to avoid harmful symmetry increases, and show that under standard assumptions, almost-isovariant maps are generic in expressive ENN families. Experiments on synthetic k‑fold structures and the QM9 dataset validate the theoretical predictions.

## Strengths
- **Theoretical rigor and novelty**: The paper establishes a comprehensive mathematical foundation for symmetry increase, introducing the symmetry infimum (Theorem 3.1) and proving its uniqueness. This formalizes and generalizes prior empirical observations and provides a predictive framework beyond existing special‑case analyses.
- **Practical algorithms and guidelines**: The authors develop computable algorithms (Algorithms 1‑3) for determining the symmetry infimum and offer concrete design guidelines (§4.2) to prevent unwanted symmetry increases in ENN feature spaces, effectively bridging theory and practice.
- **Strong empirical validation**: Experiments on synthetic k‑fold structures and the QM9 dataset consistently confirm theoretical predictions. The clear binary patterns in Figure 5 and the degradation patterns in QM9 (Figures 6‑7) demonstrate the real‑world relevance of the theory.

## Weaknesses
### Major
- **Insufficient empirical validation of practical utility**: The paper claims to provide “practical guidelines for feature design to prevent harmful symmetry increases,” but the experiments do not demonstrate that following these guidelines leads to improved performance over standard ENN design practices. The QM9 experiment only shows that features undergoing full degeneration are harmful, but it does not compare a model built strictly according to the guidelines (e.g., by actively selecting features that avoid full degeneration) against a baseline that uses a standard feature set (e.g., all degrees up to *L*). Without such a comparison, the claim of practical utility remains unsupported.
- **Actionability of guidelines limited by need for prior symmetry knowledge**: The guidelines in §4.2 require knowing the orbit types (symmetries) present in the input data. In many real‑world applications, these symmetries are not known a priori, and the guidelines are presented as manual design rules rather than an automated procedure that can be integrated into ENN training. This limits the framework’s direct applicability in settings where input symmetries are unknown or vary across examples.

### Minor
- **High‑multiplicity assumption may not hold in practice**: The theoretical guarantees (e.g., Proposition 4.2) rely on high‑multiplicity representations (multiplicity *r* > dim *G*), while many practical ENNs use small channel counts (e.g., *r* = 1–16). The paper notes that predictions coincide for single‑channel cases in the examples considered (§4.1), but this equivalence is not proven in general, leaving a gap between theory and common implementations.
- **Limited discussion of computational complexity and scalability**: The algorithms for computing the symmetry infimum (Algorithm 2) require enumerating supergroups, but the paper does not analyze their runtime complexity or scalability for more complex groups or higher‑dimensional representations. This omission makes it difficult to assess the feasibility of applying the methods in large‑scale settings.
- **Experiments could be more diverse**: The experimental validation is primarily on synthetic k‑fold structures and a single real‑world dataset (QM9) with one property (isotropic polarizability). Broader evaluation on additional tasks (e.g., energy or force prediction) and datasets (e.g., MD17, OC20) would strengthen the claim of general applicability.
- **Assumptions like the manifold hypothesis are not empirically validated**: The density results (§5) assume the manifold hypothesis and other regularity conditions. The paper does not discuss the practical validity of these assumptions for real‑world data or their potential limitations, which could affect the relevance of the theoretical guarantees.
- **Technical depth may hinder accessibility**: The heavy reliance on group theory and differential geometry (e.g., orbit types, stratifications, isotropy subgroups) makes the paper challenging for a general machine‑learning audience without a specialized background. More intuitive explanations or worked examples could improve readability.

### Trivial
- *None*

## Nice-to-Haves
- Provide a complexity analysis of the proposed algorithms and discuss their scalability.
- Extend experiments to more diverse benchmarks and include ablation studies comparing feature selection based on the symmetry infimum versus arbitrary choices.
- Develop a method to enforce isovariance during training (e.g., via regularization or architectural modifications) rather than relying solely on passive design guidelines.
- Include more intuitive explanations or visualizations to make the abstract concepts (e.g., orbit types, fixed‑point subspaces) more accessible.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **“Limited architectural diversity in experiments”**: The paper uses two well‑known ENN architectures (TFN and HEGNN), which is sufficient to validate the theoretical predictions. Requesting more architectures is a generic “more models” demand that does not invalidate the core claims.
- **“Mixed experimental results lack thorough explanation”**: The paper does explain the correlation between symmetry increase and prediction error in §6.3 and §F.3.2, providing case studies for specific point groups. The explanation is adequate for the paper’s scope.
- **“Dense presentation with limited intuitive explanations”**: While the paper is mathematically dense, this is a matter of style rather than a substantive flaw. The paper includes definitions, examples, and visualizations to aid understanding, meeting the standards for a theoretical contribution.
- **“Incomplete documentation”**: The paper includes a reproducibility statement, provides code, and details experimental settings in the appendix, which is sufficient for reproducibility. Requests for hyperparameter lists or complete logs are impractical for a conference paper.

## Suggestions
- To strengthen the practical contribution, conduct an explicit comparison on QM9 (or another benchmark) between a model built following the proposed guidelines and a standard baseline that uses all available features. This would directly demonstrate the performance benefit of avoiding harmful symmetry increases.
- Discuss how the framework might be extended to settings where input symmetries are unknown, e.g., by estimating symmetry groups from data or developing adaptive feature‑selection mechanisms.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
