Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final consolidated review.

## Summary

The paper proposes a zero-shot method for detecting AI-generated images by analyzing manifold biases of pre-trained diffusion models. Through score-function analysis and Gauss's theorem, the authors derive a three-term criterion (curvature κ, gradient magnitude D, and bias ⟨b₀, x₀⟩) that characterizes stable points on the learned probability manifold—points where generated images are expected to reside. The criterion is estimated via spherical perturbations and noise predictions from a single pre-trained diffusion model (SD v1.4), mapped into CLIP embedding space with cosine similarity. The method achieves 0.835 AUC across 20 generative techniques, substantially outperforming existing zero-shot baselines, and extends effectively to few-shot settings via mixture-of-experts.

## Strengths

- **Novel conceptual approach**: Using diffusion model score functions and manifold geometry for zero-shot detection of generated images is genuinely new. The insight that generated images lie at stable local maxima on the learned probability manifold—and that this can be detected without any generated training data—is compelling and well-motivated by the toy example in Figure 2.

- **Clean mathematical insight via Gauss's theorem**: Converting volume integrals (curvature) to surface integrals via the divergence theorem (Claim 1, Eq. 13–14), enabling Monte Carlo estimation from spherical perturbation samples, is an elegant and non-trivial mathematical contribution. The derivation connecting κ(x₀) and D(x₀) to inner products of computable quantities (u_d, h) is well-constructed.

- **Validated curvature estimator**: Figure 3c–d empirically demonstrates that the κ estimator is unbiased and consistent, with convergence confirmed in log-log plots. This gives confidence that the mathematical framework translates to reliable computation.

- **Comprehensive empirical evaluation across 20 generative models**: The evaluation spans GANs, diffusion models, and commercial tools (Midjourney, DALL-E), which is more thorough than most prior work. The method achieves strong absolute performance (0.835 AUC) and generalizes across all three technique categories (GANs: 0.85, Diffusion: 0.85, Commercial: 0.79 in Fig. 5).

- **Practical robustness**: Table 2 demonstrates stable performance across perturbation counts (AUC 0.828–0.835 for s=4–64), noise levels, different base diffusion models (SD v1.4/v2, Kandinsky: 0.826–0.835), and image corruptions (JPEG: 0.79, Gaussian blur: 0.822), indicating the method is not brittle.

- **Effective few-shot extension**: The MoE framework (Fig. 6) consistently improves over standalone methods (0.85–0.88 AUC with 100–100K samples vs. 0.77–0.795 for baselines), demonstrating practical value.

## Weaknesses

### Fatal
None.

### Major

- **Theory-practice disconnect via CLIP mapping**: The entire theoretical framework (Claims 1–3, Eqs. 7–18) is derived for the score function ∇log p and its geometric properties (curvature κ, gradient magnitude D) defined via Euclidean inner products in the original data/latent space. However, Section 4.3 explicitly states: "We map u_d, h, x₀ to CLIP before calculating C(x₀). In CLIP space, cosine similarity is used as the correct way to multiply embeddings." There is no justification for why manifold curvature, divergence, or gradient magnitude—quantities defined via differential geometry on a specific manifold—should be preserved or meaningful after a nonlinear, dimensionality-reducing projection into CLIP space, or why cosine similarity should replace Euclidean inner products. This is not a minor implementation detail; it fundamentally changes the space in which the geometric quantities are computed. The theory effectively motivates the *form* of the criterion (normalized score function dotted with u_d, h, and x₀) but does not justify the actual computation. The paper's central claim of "theoretical grounding" and that the criterion "approximates manifold-bias criteria" (Section 4.3, Important take-away) is therefore overstated. Without an experiment computing the criterion in the original space (pixel or latent), it is impossible to determine whether the geometric structure the theory predicts is what drives detection, or whether the method works for reasons entirely unrelated to manifold geometry.

- **Baselines performing below chance without discussion**: In Table 1, AEROBLADE achieves AUC 0.444 and RIGID achieves 0.439—both below random chance (0.5). An AUC below 0.5 means the criterion is systematically inverted. Figure 5 further shows RIGID at 0.38 on GANs and 0.40 on diffusion models, and AEROBLADE at 0.37 on diffusion models. The paper never discusses this anomaly. Below-chance AUC can indicate incorrect implementation, inappropriate evaluation setup, or that the baselines are fundamentally unsuited to this benchmark. In any case, beating baselines that perform below chance is uninformative about the proposed method's quality relative to what is achievable. While the absolute 0.835 AUC is still meaningful, the claimed "significant margin" improvement (Section 5.2) is misleading without addressing why baselines fail so catastrophically.

- **Missing ablation of individual criterion terms (κ, D, ⟨b₀, x₀⟩)**: Table 2 varies hyperparameters (S, α, base model, corruption) but never ablates the three components of the criterion individually. The theory assigns distinct geometric meanings to curvature (κ), gradient magnitude (D), and bias (⟨b₀, x₀⟩), yet there is no experiment showing what happens with only κ, only D, only ⟨b₀, x₀⟩, or pairwise combinations. This is essential for (a) understanding which component drives performance, and (b) evaluating whether the theoretical predictions (curvature should be high and gradient should be low for generated images) are actually borne out in practice. Without this, the connection between theory and empirical success remains unsubstantiated.

### Minor

- **Cross-model generalization lacks controlled investigation**: Section 6 acknowledges there is "no comprehensive theory to explain" why a criterion derived from SD v1.4 detects images from GANs and commercial tools. The paper hypothesizes shared training data but provides no experiment controlling for this (e.g., testing on a diffusion model trained on a substantially different dataset). This is an important open question, though the paper is transparent about it.

- **a = b = c = 1 without justification**: The three terms (κ, D, ⟨b₀, x₀⟩) have different scales and units; setting their weights equal is a significant design choice. While the paper notes "tuning is possible" and common factors are absorbed, no justification or ablation is provided for this specific choice.

- **No analysis of detection on SD 1.4-generated images specifically**: If the theory is correct, detection should be strongest for images from the analyzed diffusion model (SD v1.4). This straightforward and informative experiment is absent, which would have helped substantiate the theory and distinguish manifold-based detection from generic real/fake separation.

### Trivial
None.

## Nice-to-Haves

- A CLIP-only baseline: compute a simple criterion (e.g., CLIP embedding similarity between x₀ and the denoised prediction x̂₀) without the score-function geometric formulation, to test whether the geometric framework adds value beyond CLIP's own real/fake discriminability.
- Computing the criterion in pixel/latent space (without CLIP mapping) to directly test whether the geometric structure predicted by the theory drives detection performance.
- Per-model failure analysis for the few techniques where the method performs poorly (visible in Fig. 5b polar plot).

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Mitchell et al. analogy is "misleading"**: The harsh critic claims the analogy to Mitchell et al. (2023) is misleading because Mitchell et al. use explicit log-probability in the same space while this paper computes in a different space. While the CLIP mapping is a real issue (addressed above), the analogy is about the conceptual approach (using curvature of the learned probability manifold for detection), not about identical implementation. The paper does not claim the approaches are parallel in implementation. Removed as overstated.

- **Confusing presentation of ∇log p_α(x) = -(1/√α)ε derivation**: The critic notes the presentation "is not quite right" because replacing x̂₀ with x₀ gives -(1/α)(x_t - √(1-α)x₀), which equals -(1/√α)ε only because x_t = √(1-α)x₀ + √α ε. The paper's statement is mathematically correct—the two expressions are indeed equivalent. This is a presentation nitpick, not an error. Removed as trivial.

- **Corollary 2 is "circular"**: The critic claims the approximation E[∇log p/‖∇log p‖] ≈ 0 is circular because it "assumes the very condition the method is trying to detect." This misreads the derivation. The approximation holds because on ∂B₀, the score function approximates -(1/√α)u_d (by Eq. 5 and the concentration of measure), and averaging normalized u_d over the sphere gives approximately zero. This is an approximation whose quality varies with the accuracy of the score function on the sphere—it is not assuming the conclusion. Removed as factually wrong.

- **Data leakage in MoE**: The critic claims the 1K labeled samples for MoE training constitute data leakage. The paper explicitly states "these where randomly selected in an additional train-test split, implemented on the dataset initially used for zero-shot testing." This is standard few-shot evaluation practice—using a held-out portion of the test distribution for training. The few-shot results are not claimed to be zero-shot. Removed as misunderstanding of evaluation protocol.

- **Concentration of measure argument doesn't justify uniform distribution on sphere**: The critic argues that x_t being concentrated near the sphere doesn't mean the score function evaluated on a uniform distribution on that sphere is a good approximation. The paper's argument is that u_d is uniformly distributed on ∂B₀ by construction (Eq. 10), and because in high dimensions ε ≈ u_d (concentration of measure, Fig. 4b-c), the diffusion model—which was trained to denoise x_t—will perform well on x̃ constructed from u_d. This is a reasonable argument. Removed as overstated criticism.

- **Missing related works**: Removed per hard rules—cannot verify existence of uncited works.

- **Formatting/style issues**: Removed per hard rules.

## Novel Insights

The most insightful observation across the reviews is that the paper's discriminative power may rely on the *failure* of the approximation in Corollary 2 for real images, rather than its success for generated images. The theory derives that the three-term criterion captures manifold geometry for generated images, but it is equally important that real images—lying outside the diffusion model's manifold—produce qualitatively different criterion values because the score function does not approximate uniform spherical noise around them. This asymmetry is what makes detection work, but the theory does not explicitly characterize it, leaving the mechanism of detection under-theorized.

## Suggestions

- **Ablate the three criterion terms**: Run experiments with each term (κ, D, ⟨b₀, x₀⟩) individually and in pairs. This is the single most important missing experiment—it directly tests the theoretical claims and would either strengthen them substantially or reveal that the method's power comes from a different source than claimed.

- **Compute the criterion without CLIP**: Even a small-scale experiment computing the criterion using Euclidean inner products in pixel or latent space (without the CLIP mapping) would reveal whether the geometric structure drives detection or CLIP does. This would settle the theory-practice question.

- **Address the below-chance baselines**: Either explain why the baselines perform below chance (e.g., criterion inversion, dataset mismatch) or demonstrate that the comparison is fair (e.g., by flipping the baseline criteria to show their best-case performance).

## Evaluation

**Originality**: The approach of using diffusion model score functions for zero-shot detection is novel. The mathematical framework connecting curvature and gradient to computable quantities via Gauss's theorem is a genuine contribution. However, the CLIP mapping breaks the chain from theory to practice, reducing the originality of what is actually implemented to a well-motivated heuristic.

**Importance of research question**: Highly important—zero-shot detection of generated images is a pressing practical problem with limited existing solutions.

**Claim support**: The empirical claim (strong detection performance) is well-supported. The theoretical claim ("theoretical grounding") is overstated given the CLIP mapping disconnect and missing ablations.

**Soundness of experiments**: Adequate in scope (20 models) but weakened by below-chance baselines, missing term ablations, and no CLIP-free control.

**Clarity**: Generally clear, with effective toy examples and figures. The mathematical presentation is well-organized.

**Value to community**: The method sets a new benchmark for zero-shot detection and the mathematical framework provides a foundation for further work, even if the current implementation does not fully deliver on the theoretical promise.

## Calibration

**Anchors used:**

1. **High (>7)**: ANvmVS2Yr0 — "Geometry-adaptive harmonic representations in diffusion models" (avg 8.5, oral). Tightly connected theory-practice, rigorous spectral analysis directly validated. This paper is significantly stronger in theory-practice alignment.

2. **High (>7)**: 84n3UwkH7b — "Detecting, Explaining, and Mitigating Memorization in Diffusion Models" (avg 8.0, oral). Simple but rigorously connected method-theory, clean detection criterion directly derived from model behavior. More focused and better-validated contribution.

3. **Medium (4-6)**: lwn5fbqf74 — "Training-free Detection of AI-generated Images via HFI" (avg 5.5, withdrawn). Very similar topic (zero-shot detection using diffusion model components), similar concerns about technical contribution depth. Our paper has more mathematical depth but also more severe theory-practice disconnect.

4. **Medium (4-6)**: fPBExgC1m9 — "Leveraging Natural Frequency Deviation for Diffusion-Generated Image Detection" (avg 4.5, withdrawn). Similar topic, criticized for unjustified methodology and near-perfect unexplained results. Our paper is stronger—has a theoretical framework and more transparent evaluation.

5. **Medium (4-6)**: PJjHILiQHC — "Spectral Dynamics of Weights" (avg 6.25, reject). Broad empirical observations with theoretical framing, criticized for theory-practice disconnect. Our paper has a more specific and testable theoretical framework but similar gap.

6. **Low (<3)**: a8XwgTZzE0 — "Grokking through Dynamical Systems" (avg 2.0, reject). Poorly connected theory and implementation, unclear math. Our paper is far above this—clear derivations, working method, comprehensive evaluation.

This paper falls above the low anchors (clear math, working method, real results) but below the high anchors (theory-practice disconnect, missing ablations). Among medium anchors, it is comparable to the HFI paper (5.5) but with more mathematical depth, placing it around the upper end of the medium range. The below-chance baselines and missing ablations are significant but the absolute performance and breadth of evaluation are genuine contributions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>