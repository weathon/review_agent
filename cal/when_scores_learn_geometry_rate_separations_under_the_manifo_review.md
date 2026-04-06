=== CALIBRATION EXAMPLE 78 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is evocative and aligns with the core idea. The abstract succinctly presents the central claim of a rate separation (Θ(σ⁻²) for geometry vs. Θ(1) for density) and the proposed paradigm shift toward geometric learning. However, the claim that the success of score-based models "arises from implicitly learning the **data manifold**" is presented as a definitive alternative explanation, while the paper primarily provides theoretical evidence that such geometric learning is *easier* and can be achieved with weaker guarantees. The abstract should more carefully phrase this as a *plausible explanation* or a *consequence* of the theory, rather than an established fact.

### Introduction & Motivation
The introduction effectively sets up the challenge of low-noise score estimation and the manifold hypothesis. The contributions are clearly listed. The related work section (1.1) adequately surveys prior art but could be more precise in delineating the novelty. For instance, the statement that prior works "do not explicitly isolate the higher-order terms involving *p_data*" is accurate, but the claim that they "do not characterize the separation between geometry and density" might be too strong, as some works (e.g., Lu et al., 2023; Lyu et al., 2025) analyze the asymptotic structure of the score, which implicitly contains this separation. The distinction should be that this paper *quantifies* the separation in terms of error tolerance (o(σ⁻²) vs. o(1)), which is novel.

### Method / Approach
**Sections 2-3 (Preliminaries and Central Insight):** The setup is standard and clearly presented. Theorem 3.1 (informal) is the heart of the paper, and the expansion in Equation (6) elegantly shows the separation. However, the formal statement (Theorem B.2 in the appendix) should be referenced in the main text with a summary of its technical conditions (C⁴ manifold, C¹ density, etc.) to ensure transparency about the assumptions required for the expansion.

**Section 4 (Scale Separation in Existing Generative Learning):** Theorem 4.1 is a clear manifestation of the rate separation. The assumptions (Assumption 4.1) are reasonable but non-trivial (e.g., uniform rectifiable path-connectedness of the compact set *K*). The proof strategy is sound. The major limitation here, acknowledged in Section 8, is that the result is framed in terms of the score error of the *final generated distribution* π_σ. In practical diffusion models, the score is used iteratively in a reverse process, and errors accumulate along the trajectory. The theorem does not address how an o(σ⁻²) pointwise error in the *learned score network* translates to the error *E_σ* of the *generated distribution*. This gap significantly weakens the direct practical implication for diffusion models. The claim that "this insight provides a potential new explanation for the remarkable success of diffusion models" is therefore speculative without a analysis of error propagation.

**Section 5 (New Paradigm of Geometric Learning):** The tempered score (TS) Langevin dynamics is a simple and interesting idea.
- Theorem 5.1 (gradient case) is straightforward and clearly demonstrates the benefit: uniform sampling with o(σ⁻²) error.
- Theorem 5.2 (non-gradient case) is the most technically advanced contribution. However, it relies on several strong and non-standard assumptions:
    1.  **L∞ Score Error (Equation (9)):** The requirement of *uniform* o(σ^β) error is stringent. Practical score matching minimizes an L²-type loss (e.g., Fisher divergence). The authors note this as a limitation, but it remains a significant gap between theory and practice.
    2.  **Existence and Form of Stationary Distribution (Assumption B.2):** The assumption that the SDE admits a unique stationary distribution which *locally admits a WKB form* is crucial. For a general non-gradient drift, existence and uniqueness of a stationary distribution are not guaranteed. The WKB ansatz is an *ansatz*; its validity needs justification, especially when the drift is not a gradient field. The authors provide a derivation in Appendix B.4 assuming the ansatz holds, but they do not provide sufficient conditions under which it *does* hold for their specific SDE. This makes the theoretical guarantee feel conditional.
    3.  **Smoothness of c₀:** The application of the strong maximum principle to conclude c₀ is constant requires c₀ to be C² on the manifold. It is unclear from the assumptions (p_data ∈ C² is assumed, but this pertains to the data density, not the prefactor from the WKB expansion) that this smoothness holds. This step needs more justification.
    Overall, while the technical machinery is impressive, the foundational assumptions for the non-gradient case are not fully substantiated, which undermines the robustness of the main claim.

**Section 6 (Bayesian Inverse Problems):** Theorem 6.1 is a direct application of the previous results. The connection to classifier-free guidance is a nice observation, but its practical utility is only briefly explored in experiments.

### Experiments & Results
The experimental validation is preliminary and does not fully substantiate the theoretical claims, especially for large-scale models.

1.  **Synthetic Experiments (Ellipse/Circle):** These are clean and demonstrate the phenomenon on a toy example with a known manifold and controlled score error. They support Theorems 4.1 and 5.1 well.
2.  **Image Generation with Stable Diffusion:**
    - **Metrics:** Relying solely on CLIP-based metrics (P-sim and I-sim) is insufficient. These measure alignment with a prompt and pairwise image similarity, but they do not directly measure proximity to the *data manifold* or uniformity of sampling on it. Standard generative metrics like FID, precision/recall, or a measure of coverage would be more informative. The improvement in I-sim (lower is better) is consistent with increased diversity, but it could also result from images becoming more "spread out" in CLIP space in an arbitrary way, not necessarily corresponding to a better approximation of the uniform distribution on the true image manifold.
    - **Magnitude of Improvement:** The improvements in P-sim and I-sim are modest (often less than 1%). Statistical significance testing is absent. For ICLR, it is essential to show that these improvements are not due to random variation.
    - **Validation of Uniform Sampling:** The central claim of Section 5 is that TS Langevin yields the *uniform distribution* on the manifold. The experiments do not provide evidence for this claim in the image domain. How can one verify that the samples are uniform on a complex, high-dimensional image manifold? The authors could attempt indirect validation, e.g., by showing that TS generates more diverse interpolations or explores a broader set of latent codes in a disentangled representation.
    - **Ablations:** The sensitivity analysis for α (Appendix C.4) is good, but the choice of α=1 for most experiments is not theoretically motivated (the theory allows a range). More discussion on selecting α in practice is needed.
    - **Baselines:** Comparisons are limited to DDPM and PC samplers. Comparisons with state-of-the-art samplers (e.g., DPM-Solver) would strengthen the case for the practical utility of the modification.

### Writing & Clarity
The paper is generally well-written. The main ideas are presented clearly. The appendices are lengthy but necessary. Some parts of the technical proofs (especially Appendix B.4) are dense and would benefit from more intuitive scaffolding. The notation is mostly consistent, though in Appendix D, the use of *p_θ* for a parameterized density is potentially confusing given the prior use of *p_σ*.

### Limitations & Broader Impact
Section 8 correctly lists key limitations: the simplified setting for diffusion models, the L∞ error assumption, lack of statistical sample complexity results, and preliminary experiments. These are serious limitations that temper the immediate practical impact of the work. The societal impact section is missing; a brief discussion on the potential misuse of more robust generative models or the implications of uniform sampling (e.g., for fairness) would be appropriate.

### Overall Assessment
This paper presents a novel and theoretically interesting insight: a sharp rate separation in score learning under the manifold hypothesis, leading to the proposal of "geometric learning" as a more robust objective. The tempered score Langevin dynamics is a simple and promising algorithmic idea. However, the paper has significant weaknesses that prevent it from being ready for ICLR in its current form:

1.  **Theoretical Gaps:** The analysis for the non-gradient score case (Theorem 5.2) rests on strong, unverified assumptions (existence/uniqueness of a stationary distribution with a WKB form, smoothness of the prefactor). The connection to practical diffusion models is weak, as the theory does not account for the iterative nature of the sampling process.
2.  **Insufficient Empirical Validation:** The experiments on large-scale models are preliminary. The metrics used do not directly test the core theoretical claims (manifold concentration, uniform sampling), and the improvements are small and not statistically validated. The experiments do not convincingly demonstrate the superiority or even the operationalization of the "geometric learning" paradigm in practice.

The paper has the potential to be a strong contribution if these issues are addressed. For acceptance, the authors need to:
- Provide more justification for or relax the assumptions in Theorem 5.2, perhaps by proving the validity of the WKB ansatz under specific conditions.
- Either provide a more rigorous bridge to practical diffusion models (e.g., analyzing error propagation in the reverse process) or more carefully qualify the implications of the theory.
- Substantially expand the experimental section: include standard generative metrics, perform statistical tests, provide more direct evidence for uniform sampling (e.g., on synthetic data with known manifold structure), and conduct more comprehensive comparisons and ablations.
- Address the missing discussion on societal impact.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a new perspective on score-based generative models (e.g., diffusion models) under the manifold hypothesis. The authors argue that these models succeed by implicitly learning the geometry of the data manifold, not the full distribution. Their key theoretical insight is a *rate separation*: geometric information (distance to the manifold) appears at order Θ(σ⁻²) in the low-noise limit, while distributional information (on-manifold density) appears only at order Θ(1). This suggests that learning the manifold is substantially easier than learning the exact distribution. Based on this, the paper introduces a simple "tempered score" Langevin dynamics that provably recovers the *uniform* distribution on the manifold with a much weaker score-error requirement (o(σ⁻²)) compared to exact distribution recovery (o(1)). The authors validate their theory with experiments on synthetic manifolds and image generation using Stable Diffusion.

### Strengths
1. **Novel theoretical contribution:** The rate separation between geometric and distributional information is a fresh and insightful way to interpret the success of score-based models. Theorem 3.1 and its formal versions clearly articulate this phenomenon, and the subsequent theorems rigorously explore its implications for generative modeling, uniform sampling, and Bayesian inverse problems.
2. **Practical relevance and simplicity:** The proposed tempered score (TS) Langevin dynamics is a one-line modification to existing sampling schemes (e.g., the corrector step in predictor-corrector algorithms). Experiments on Stable Diffusion show that TS can improve both diversity (lower inter-image CLIP similarity) and quality (higher prompt similarity) across multiple prompts, demonstrating potential real-world utility.
3. **Theoretical rigor:** The paper provides detailed proofs using advanced techniques such as Laplace's method, WKB asymptotics, and analysis of non-reversible SDEs. The handling of non-gradient score fields (Theorem 5.2) is particularly nontrivial and strengthens the applicability of the results.

### Weaknesses
1. **Limited empirical validation:** The experiments are preliminary. The synthetic example (ellipse/circle) is simple, and the image generation experiments are confined to a single model (Stable Diffusion 1.5) with only three prompts and limited metrics. There is no comparison to state-of-the-art diffusion samplers or extensive ablation studies. The improvements in CLIP scores, while consistent, are modest.
2. **Strong assumptions that may not fully hold in practice:** The analysis assumes a compact, boundaryless, C⁴ manifold with a strictly positive C¹ or C² data density. Real-world data manifolds may have boundaries, singularities, or less smoothness. Additionally, the theoretical results rely on L∞ bounds on the score error, which is stricter than the L²-type objectives typically used in training (e.g., denoising score matching). The paper acknowledges these limitations but does not address how violations might affect the conclusions.
3. **Incomplete treatment of diffusion models:** The theoretical results focus on the stationary distribution of Langevin dynamics at a fixed noise level, whereas practical diffusion models involve a reverse process across a continuum of noise levels. The paper does not analyze error accumulation over the reverse trajectory, which is critical for understanding actual diffusion model performance.
4. **Clarity could be improved:** The paper is dense with technical notation and asymptotic expansions. While the main ideas are well-motivated, the proofs are highly technical and may be difficult for a broad audience to follow. The connection between the theoretical results and practical algorithms could be made more explicit, especially for readers less familiar with WKB methods.

### Novelty & Significance
The paper presents a novel and significant theoretical perspective on score-based generative models. The rate separation insight provides a plausible explanation for why diffusion models often generate realistic-looking samples even with imperfect score estimates. The proposal to shift focus from distributional learning to geometric learning is thought-provoking and could inspire new research directions. The tempered score modification is simple yet powerful, offering a practical way to encourage uniform exploration of the data manifold. However, the significance is somewhat tempered by the preliminary nature of the experiments and the strong assumptions required by the theory. For ICLR, which values both theoretical and empirical contributions, the paper would benefit from more thorough experimental validation.

### Suggestions for Improvement
1. **Expand the experimental evaluation:** Test the tempered score approach on a wider range of datasets (e.g., CIFAR-10, FFHQ) and diffusion architectures (e.g., ADM, EDM). Include comparisons to other diversity-enhancing techniques (e.g., truncation, guidance tuning) and provide more comprehensive metrics (e.g., FID, precision/recall). Visual examples and user studies would also strengthen the empirical claims.
2. **Relax theoretical assumptions:** Investigate whether the L∞ score-error condition can be relaxed to an L² bound more aligned with practical training objectives. Explore extensions to manifolds with boundaries or less smoothness, perhaps through numerical simulations. An analysis of error propagation over the full reverse diffusion process would greatly enhance the relevance to real-world models.
3. **Improve clarity and accessibility:** Add more intuitive explanations of the technical results, possibly with additional figures or a less formal overview. Simplify notation where possible (e.g., by reducing the number of auxiliary functions). Clearly distinguish between the variance-exploding (VE) and variance-preserving (VP) cases in the main text, as they are often treated separately in practice.
4. **Deeper discussion of limitations and future work:** The limitations section is brief. Elaborate on the challenges of generalizing the theory to full diffusion processes, the statistical sample complexity implications, and the impact of discretization errors in practical implementations. Suggest concrete steps for addressing these issues in future research.
5. **Strengthen the related work section:** While the paper covers relevant literature, it could more clearly differentiate its contributions from prior work on diffusion models and manifold learning (e.g., how the uniform sampling approach compares to recent methods for improving diversity). Highlighting the unique aspects of the rate separation argument would help position the work.

**Overall, the paper presents a compelling theoretical insight with promising practical implications. However, to meet ICLR's high standards, it needs more extensive empirical validation and a more thorough discussion of its limitations and applicability to real-world models.**

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Controlled validation of the rate separation claim.** The paper lacks an experiment where the score error is systematically varied (e.g., by adding controlled noise of known magnitude) and the resulting distribution is measured as σ→0. Without this, the core theoretical claim—that o(σ⁻²) error yields manifold concentration while o(1) error is needed for the true density—remains unverified.
2. **Comparison with explicit manifold-learning baselines.** The paper advocates a shift to geometric learning but does not compare against standard manifold-learning methods (e.g., VAEs, GANs with low-dimensional latent space) on tasks like uniform sampling or diversity. This omission weakens the claim that the proposed approach is novel or superior for geometric recovery.
3. **Ablation study on the tempering parameter α.** The method introduces a key hyperparameter α, yet experiments only use α=1. A systematic analysis of how α affects sample quality/diversity across different σ regimes is missing, making it unclear how to choose α in practice and whether the theoretical bounds hold empirically.
4. **Quantitative evaluation of recovered manifold geometry.** The paper claims to recover the manifold, but there is no quantitative assessment (e.g., intrinsic dimension estimation, geodesic distance preservation, or latent traversals) to verify that the geometry is correctly learned. This is essential to substantiate the geometric-learning paradigm.

### Deeper Analysis Needed (top 3-5 only)
1. **Relaxation of L∞ score-error assumption.** The theory assumes L∞-bounded score error, but practical training minimizes L²-like losses (e.g., Fisher divergence). The paper must discuss whether the rate separation persists under L² errors, as this directly affects the relevance to real-world score matching.
2. **Cumulative error analysis for full diffusion sampling.** The analysis considers a fixed σ, but diffusion models involve a reverse process with accumulating errors over time. Without extending the rate separation to the entire sampling trajectory, the implications for diffusion models remain speculative and incomplete.
3. **Justification of the WKB ansatz (Assumption B.2).** The non-gradient analysis crucially assumes the stationary distribution admits a WKB form. The paper provides no conditions under which this holds (e.g., ergodicity, smoothness), making Theorem 5.2 appear as an unsubstantiated assumption rather than a proven result.
4. **Discussion of practical σ regimes.** The theory requires σ→0, but real diffusion models use finite noise schedules. The paper should analyze how small σ must be for the rate separation to manifest and whether typical schedules (e.g., in Stable Diffusion) meet this condition.

### Visualizations & Case Studies
1. **Visual decomposition of score components.** Plotting the geometric (Θ(σ⁻²)) and distributional (Θ(1)) parts of the score as σ varies for a simple manifold would visually confirm the rate separation and show the crossover point, making the theory more tangible.
2. **Case studies of failure modes under large score errors.** Demonstrating how samples degrade when score error is O(σ⁻²) vs. O(1) would illustrate the practical consequences of the theory and help diagnose model limitations.
3. **Manifold visualizations for image data.** Using dimensionality reduction (e.g., t-SNE, UMAP) to visualize the latent structure of samples generated by standard vs. tempered score methods would provide direct evidence that TS Langevin yields more uniform coverage of the manifold.

### Obvious Next Steps
1. **Extend theory to L² score errors.** The most critical next step is to relax the L∞ assumption to L², aligning with practical training objectives. This would significantly strengthen the paper’s applicability.
2. **Analyze error propagation in the full reverse diffusion process.** Incorporating time-dependent score errors and studying their accumulation is necessary to draw concrete conclusions about diffusion models.
3. **Provide sufficient conditions for the WKB ansatz.** Either proving that Assumption B.2 holds under reasonable conditions or numerically verifying it is essential to trust the non-gradient results.
4. **Experiments on higher-dimensional synthetic and real datasets.** Testing on more complex manifolds (e.g., high-dimensional spheres) and diverse image datasets (e.g., CIFAR-10, FFHQ) would bolster the empirical claims.

# Final Consolidated Review
## Summary
This paper establishes a sharp rate separation under the manifold hypothesis: the score of a Gaussian-smoothed data distribution encodes geometric information about the manifold at order Θ(σ⁻²), while the on-manifold density appears only at order Θ(1). This insight suggests that learning the data manifold is substantially easier than learning the full distribution. The authors propose a simple "Tempered Score" (TS) Langevin dynamics which, with only o(σ⁻²) score accuracy, provably samples uniformly from the manifold—a much weaker requirement than the o(1) accuracy needed for exact distribution recovery.

## Strengths
- **Novel theoretical insight:** The paper rigorously quantifies a fundamental separation between geometric and distributional information in the low-noise score, providing a fresh and compelling lens through which to interpret the success of score-based models. The expansion in Theorem 3.1 and its consequences are clear and significant.
- **Simple and practical algorithm:** The proposed TS Langevin dynamics is a one-line modification to standard sampling schemes (e.g., the corrector step in diffusion models). Preliminary experiments on Stable Diffusion show it can improve both diversity (lower inter-image CLIP similarity) and quality (higher prompt similarity), demonstrating potential real-world utility.

## Weaknesses
- **Theoretical gaps in the non-gradient case:** The central result for general score estimators (Theorem 5.2) relies on strong, unverified assumptions. It assumes the SDE admits a unique stationary distribution that locally satisfies a specific WKB ansatz (Assumption B.2), and it requires the prefactor in this ansatz to be sufficiently smooth to apply the strong maximum principle. The paper does not provide sufficient conditions under which these assumptions hold, making the guarantee feel conditional rather than proven.
- **Limited empirical validation of core claims:** The experiments, while promising, are preliminary. The synthetic example (ellipse/circle) validates the theory in a controlled setting, but the image generation experiments do not directly test the paper's central claim—that TS Langevin recovers the *uniform distribution* on the manifold. The exclusive use of CLIP-based metrics (prompt and inter-image similarity) does not measure uniformity or manifold fidelity. Improvements are modest and statistical significance is not established.
- **Incomplete bridge to practical diffusion models:** The analysis focuses on the stationary distribution of a Langevin sampler at a fixed noise level σ. Practical diffusion models involve a reverse process across a continuum of noise levels, where errors accumulate. The paper does not analyze this error propagation, leaving the direct implications for diffusion model performance speculative.

## Nice-to-Haves
- A more comprehensive experimental evaluation on diverse datasets (beyond three prompts) with standard generative metrics (e.g., FID, precision/recall) and comparisons to state-of-the-art samplers.
- An investigation into whether the stringent L∞ score-error assumption can be relaxed to an L²-type bound more aligned with practical training objectives like denoising score matching.
- A discussion of societal impact, given the work aims to improve the robustness and diversity of generative models.

## Novel Insights
The paper's key novel insight is the explicit quantification of a rate separation: geometric information (distance to the manifold) dominates the score at order Θ(σ⁻²), while distributional details appear only at order Θ(1). This explains why approximate scores can still force samples onto a plausible-looking manifold while failing to recover the true data density. It motivates a paradigm shift from the hard task of distributional learning to the more attainable goal of geometric learning, exemplified by the efficient recovery of the uniform manifold measure—a structured and useful object—with relaxed score accuracy.

## Suggestions
- Provide more justification for the foundational assumptions in Theorem 5.2, such as proving the validity of the WKB ansatz under specific, verifiable conditions related to the learned score field.
- Design experiments that more directly test the claim of uniform manifold sampling, even if indirectly (e.g., by analyzing the coverage of a known latent space or using metrics sensitive to distributional uniformity on synthetic manifolds with ground truth).

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
