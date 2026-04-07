## Summary
KOALA introduces a novel adversarial detector that flags inputs as attacked when predictions from two complementary similarity metrics—KL divergence (sensitive to dense, low-amplitude perturbations) and a custom L0-based score (sensitive to sparse, high-impact changes)—disagree. The method requires only lightweight fine-tuning on clean images to align embeddings and is accompanied by a formal detection guarantee under a set of assumptions and a sufficient condition on prototype separation.

## Strengths
- **Novel detection principle**: The core idea of forcing detection via disagreement between two geometrically motivated, complementary metrics (KL and L₀) is conceptually fresh and well-motivated by an analysis of perturbation types. This offers a new perspective beyond single-metric or semantics-driven detectors.
- **Theoretical grounding**: The paper provides a formal theorem (Theorem 1) with a detailed proof (Appendix B) that guarantees detection when a sufficient "coordinate gap" exists between class prototypes. This effort to provide rigorous guarantees is rare and valuable in the empirical landscape of adversarial detection.
- **Practical and lightweight design**: KOALA operates as a plug-in detector without adversarial training, architectural changes, or semantic priors. The required fine-tuning uses only clean images and a composite loss, making it easily deployable.

## Weaknesses
- **Theoretical assumptions limit practical guarantees**: The theorem’s guarantees rely on strong assumptions (A1–A3), particularly A3 (coordinate-wise perturbation bound |δ_i| ≤ (3/2)|p_i*|) and the implicit reliance on a Lipschitz constant to link pixel-space and feature-space bounds (A2). These are not empirically validated and may not hold broadly, making the theoretical guarantee conditional and less applicable to real-world deployments.
- **Unclear operationalization of “theorem-compliant” criterion**: Experiment 1 splits data based on whether “the sufficient inter-class prototype separation” holds, but the paper does not specify the threshold Γ_i(ϵ) or the exact procedure for determining compliance. This lack of reproducibility undermines the empirical validation of the core theorem.
- **Missing comparison to state-of-the-art detectors**: The evaluation compares KOALA only to ablations of itself (different metric combinations). Without benchmarking against established detection methods (e.g., LID, Mahalanobis, feature squeezing), it is impossible to assess its relative performance and contribution to the field.
- **Limited attack evaluation**: Experiments are confined to ℓ∞-bounded attacks (PGD, CW, AutoAttack). The detector’s efficacy under other threat models (e.g., ℓ₂, ℓ₁) or against adaptive attacks specifically designed to evade the two-metric disagreement is unexplored, leaving its general robustness in question.

## Nice-to-Haves
- Sensitivity analysis for hyperparameters τ (L₀ threshold) and ϕ (smoothing parameter), with guidance on setting them.
- Extension to additional architectures (e.g., Vision Transformers) and larger-scale datasets (e.g., full ImageNet) to further support the “plug-and-play” claim.
- Visualization of feature-space perturbations or prototype/embedding structures to illustrate the dense vs. sparse attack patterns and the effect of fine-tuning.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Strength**: “The paper is well-written” – generic.
- **Weakness**: “The L₀ metric definition is unconventional” – the paper clearly defines it as a sparse-change detector; the design choice is justified.
- **Weakness**: “The proof is extremely long and dense” – not a substantive critique; detailed proofs are appropriate for an appendix.
- **Weakness**: “Missing a dedicated limitations section” – while a limitations discussion would strengthen the paper, its absence is not a core flaw; the assumptions and non-compliant results implicitly highlight limitations.
- **Weakness**: “The loss weights ω_L₀=0.9, ω_KL=0.1 are not justified” – the paper notes L₀ is harder to optimize; an ablation would be nice but is not essential.

## Novel Insights
The paper’s key novel insight is that adversarial perturbations under an energy budget tend to manifest as either dense, low-amplitude shifts or sparse, high-impact changes, and that these two types can be captured by two complementary metrics (KL divergence and an L₀-based score). By forcing detection when predictions from these metrics disagree, the method creates a mutually exclusive condition that can be formally guaranteed under certain geometric separations in the embedding space. This geometric perspective on detection is a distinct contribution beyond purely empirical or semantics-driven approaches.

## Suggestions
- Clearly define the operational criterion (e.g., the threshold Γ_i(ϵ)) used to split “theorem-compliant” and “non-compliant” samples in Experiment 1, ensuring reproducibility.
- Add a comparative evaluation against state-of-the-art adversarial detectors (e.g., LID, Mahalanobis, feature squeezing) on the same benchmarks to establish KOALA’s relative performance.
- Evaluate the detector under additional threat models (ℓ₂, ℓ₁ norms) and consider testing against an adaptive attacker aware of the two-metric disagreement mechanism.