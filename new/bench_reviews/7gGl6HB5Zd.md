Now I have read the full paper. Let me produce the final consolidated review.

---

## Summary

This paper introduces a zero-shot framework for detecting AI-generated images by analyzing biases in the implicit probability manifold of a pre-trained diffusion model. Using score-function analysis, the authors derive curvature (κ), gradient magnitude (D), and a denoiser-bias term (⟨b₀, x₀⟩), combining them into a detection criterion C(x₀). The method achieves substantially better zero-shot AUC (0.835) than the two existing zero-shot baselines (~0.44 each) across 20 generative techniques, and is also shown to boost few-shot detection when combined with an existing few-shot detector.

---

## Claims and Support

**Claim 1 – C(x₀) is a theoretically grounded criterion based on manifold geometry (curvature, gradient, denoiser bias).**
*Partially supported.* The derivations in Claims 1–3 and Corollaries 2–3 are mathematically coherent *in Euclidean / image space*. However, Sec. 4.3 explicitly states: "We map u_d, h, x₀ to CLIP before calculating C(x₀)... In CLIP space, cosine similarity is used as the correct way to multiply embeddings." This remapping is not covered by the derivations. The paper says C(x₀) *approximates* the manifold-bias criteria, but does not prove that the cosine-similarity/CLIP-projection preserves the curvature or gradient ordering claimed. The claim is thus validated as a useful heuristic with theoretical motivation, not as a strict derived estimator.

**Claim 2 – First zero-shot analysis of a pre-trained diffusion model for generated image detection.**
*Well-supported.* The related-work review identifies only two prior zero-shot approaches (AEROBLADE and RIGID), neither of which uses diffusion-model manifold analysis. This novelty claim is credible.

**Claim 3 – The method has theoretical grounding lacking in prior zero-shot methods.**
*Partially supported.* The paper provides genuine mathematical derivations. But the theoretical guarantee does not formally extend to the deployed CLIP-space criterion. The claim holds as "theory-inspired" rather than "theory-grounded" in the strict sense.

**Claim 4 – The method outperforms current zero-shot approaches on 20 unseen generative techniques.**
*Well-supported* within the scope of compared methods. Margins are large (AUC 0.835 vs. 0.444/0.439) across GANs, diffusion models, and commercial tools. The two compared methods are, per the paper's own review, the only existing zero-shot baselines.

**Claim 5 – Excellent generalization to unseen generative techniques.**
*Mostly supported empirically; theoretically unexplained.* The paper's limitations section honestly states: "there is no comprehensive theory to explain it." This is an acknowledged gap, not a concealed one.

**Claim 6 – Extension to few-shot regime via mixture-of-experts.**
*Partially supported.* Fig. 6 shows the proposed zero-shot score consistently helps an existing few-shot detector when added as a feature. However, calling a random forest over two pre-computed scores a "mixture-of-experts methodology" overstates what is actually a feature complementarity experiment.

**Claim 7 – Robustness across backbones, noise levels, and corruptions.**
*Modestly supported.* Table 2 shows small AUC variations across S, α, and two corruption types. Robustness under adversarial settings and wider corruption families is not demonstrated.

---

## Strengths

- **Strong empirical zero-shot performance.** AUC 0.835 versus 0.444/0.439 for the two existing zero-shot baselines is a large, consistent margin across GANs, diffusion models, and commercial tools. This is the most convincing empirical contribution of the paper.
- **Novel theoretical framing.** The paper is the first to connect diffusion model score functions and manifold geometry (TV-curvature, divergence theorem) to generated-image detection. Even if the CLIP-space implementation does not perfectly track the derived quantities, the conceptual framework is genuinely original.
- **Cross-family generalization without retraining.** The method is built on SD 1.4 yet works across GAN and commercial-tool outputs, demonstrating a practically useful property.
- **Comprehensive benchmark.** ~200K images from 20 generators spanning three distinct families is a genuine stress test for zero-shot generalization.
- **Complementary few-shot signal.** The scatter plot in Fig. 6 shows the proposed criterion is decorrelated from Cozzolino et al. (2024a), providing a real and useful complementary signal for ensemble methods.
- **Honest limitations section.** The paper explicitly admits the lack of theoretical explanation for cross-model generalization—a candid acknowledgment uncommon in this literature.

---

## Weaknesses

### Fatal
*None.* The core empirical contribution is sound and the theoretical motivation, while imperfect, does not introduce fundamentally incorrect reasoning.

### Major

- **Theory-practice gap via CLIP projection.** The derivations in Equations (12)–(18) use Euclidean inner products in the data/image space. The implemented criterion in Sec. 4.3 maps all quantities to CLIP embeddings and replaces inner products with cosine similarities. No argument is provided that curvature, gradient magnitude, or the denoiser-bias term preserve their relative ordering under this nonlinear projection. This means C(x₀) is best understood as a CLIP-space heuristic *inspired by* the derived quantities, not as a faithful estimator of κ, D, and ⟨b₀, x₀⟩. The paper frames this as an "approximation," but does not provide evidence—e.g., correlation studies between pixel-space and CLIP-space criterion values—that the approximation is valid. This gap weakens the central "theoretically grounded" claim.

- **Insufficient justification of the key approximation in Eq. (17).** The step E_{x~x̃|x₀}[∇log p_α(x) / ‖∇log p_α(x)‖₂] ≈ 0 is justified by appealing to "integration of normals over the sphere is zero, and ∇log p_α approximates the uniform spherical noise." But the normalized score function is *not* uniform on the sphere—it is the raw noise that is approximately uniform, not its normalized score-function version. This conflation is non-trivial in high dimension, and no empirical check (e.g., directly measuring the left-hand side on real and generated images) is provided.

- **Cross-model generalization is empirically strong but entirely unexplained.** The paper's most striking and practically important result—that a criterion induced by SD 1.4's manifold detects GAN outputs—is acknowledged in Sec. 6 to have "no comprehensive theory." This is the right epistemic posture, but it means that the explanatory framework in Sections 3–4 does not account for the method's most notable property. The method works, but not demonstrably for the reason the paper argues.

### Minor

- **Few-shot contribution is overstated.** Training a random forest over two scalar outputs (the proposed criterion and Cozzolino et al.'s score) is a sensible and practically useful experiment, but framing it as a "mixture-of-experts methodology" implies a methodological advance beyond what is presented. No ablation with simpler fusion rules (e.g., logistic regression, linear combination) is shown to establish that a random forest is necessary, and no variance across random 1K training splits is reported.

- **Computational cost is not reported.** The method requires s=64 forward passes through a full diffusion model per image, plus LLaVA caption generation and CLIP encoding. No inference time comparison with AEROBLADE (single AE pass) or RIGID is provided. This is relevant for practical deployment and should be disclosed.

- **Caption dependency under-analyzed.** The method uses LLaVA-generated captions as input to SD 1.4, which is text-conditioned. The effect of caption quality or variability on C(x₀) is not analyzed, despite its potential to affect the criterion non-trivially.

- **Robustness evaluation is narrow.** Only JPEG compression and Gaussian blur (one severity each) are tested. Real-world adversarial settings involve platform-specific recompression chains, stronger compression, resizing, and denoising. The 3.45% AUC drop under JPEG is noted but no mitigations are offered.

- **Threshold calibration sensitivity is unexamined.** The threshold is calibrated using 1K real images. The sensitivity of classification performance to the choice and diversity of this calibration set is not analyzed—relevant for zero-shot deployment scenarios where calibration data may differ from the test domain.

### Trivial

- **a = b = c = 1 weight selection has no principled basis.** The paper notes "tuning is possible" but provides no ablation varying these weights, leaving the relative importance of κ, D, and ⟨b₀, x₀⟩ unknown. Even a 3×3 grid search would clarify this.

---

## Nice-to-Haves

- Evaluate C(x₀) on images from SD 1.4 itself (the analyzed model) to verify the core theoretical prediction that same-model generations should be the most detectable.
- Provide a correlation analysis between the pixel-space version of C(x₀) and the CLIP-space version, to empirically validate that the CLIP projection preserves the ordering implied by the theory.
- Empirically measure E[∇log p_α(x) / ‖∇log p_α(x)‖₂] on real and generated images to assess how close it is to zero, validating or constraining the step in Eq. (17).
- Show per-technique histogram distributions of C(x₀) (not just for LDM as in Fig. 4a) to clarify whether a single threshold generalizes or whether different generators require recalibration.
- Report inference time per image for all methods.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Availability / existence of cited models (AEROBLADE, RIGID, Cozzolino et al., etc.):** Multiple reviewers raise reproducibility concerns. Per the hard rules, if the paper cites a method, it exists and is available. Removed.

- **"Limited baseline comparisons" demanding DIRE, ConV, HFI:** The paper explicitly states in Sec. 2: "To the best of our knowledge, Ricker et al. (2024) and He et al. (2024) are the only such [zero-shot] methods." Comparing zero-shot methods is the proper scope. Demanding inclusion of supervised or semi-supervised baselines as competitors to a zero-shot method is scope creep (though it would be a nice-to-have to include supervised results as an upper bound). Removed as a weakness.

- **"Dependence on SD 1.4 creates a longevity / expiration date concern":** This is a speculative concern about future deployment scenarios and does not affect the validity of the current results. Removed.

- **"Top-10 accuracy in Table 1 is misleading":** The Spark reviewer notes that RIGID Top-10, AEROBLADE Top-10, and Ours Top-10 use different per-method subsets for the accuracy column, making direct comparison in those columns difficult. This is a valid observation about Table 1's presentation but is minor formatting/interpretation guidance—the main metrics (overall AUC, AP, Accuracy) tell the cleaner story. Weakened to trivial/formatting.

- **Missing related works not cited by the paper:** Per the hard rules, I cannot confirm external existence of works not cited in the paper and thus do not raise this.

---

## Novel Insights

The most genuinely novel observation across the reviews is the unresolved question of *what* the CLIP-projected criterion is actually measuring. The paper derives a criterion in image/score space, then maps it to CLIP space—a feature space trained to align images with natural language descriptions. If C(x₀) works well in CLIP space, this may not be because CLIP preserves the diffusion manifold's curvature geometry, but because CLIP embeddings are themselves biased by a training distribution that systematically differs between real and generated images. This opens a deeper question: is the detection power coming from the diffusion model's manifold geometry (the theoretical claim), or from CLIP's representation biases (an orthogonal, data-driven signal)? Resolving this would substantially strengthen or reformulate the theoretical contribution.

---

## Suggestions

1. **Bridge the theory-practice gap empirically:** Compute C(x₀) in both pixel space (as the derivation implies) and CLIP space, and report their Spearman correlation across real and generated images. If the ranking is well-preserved, the CLIP mapping is justified empirically. If not, the paper's theoretical framing should be revised.

2. **Validate Eq. (17):** Directly estimate E[∇log p_α(x) / ‖∇log p_α(x)‖₂] on samples from ∂B₀ around real and generated images. This is a simple experiment that either confirms the approximation or reveals when it breaks down.

3. **Ablate the three criterion terms individually (κ, D, ⟨b₀, x₀⟩):** Show results for each term alone and in pairs. This would clarify which geometric property drives detection and whether the multi-term framing is supported by the data.

4. **Report inference time.** Even a single number (seconds per image on a standard GPU) would allow readers to assess practical applicability.

5. **Reframe the few-shot contribution more precisely.** "Our zero-shot score provides a complementary feature that improves few-shot detectors" is an accurate and still valuable claim; the "mixture-of-experts methodology" framing oversells it.

---

## Score and Decision

**Originality:** High. First application of diffusion model manifold analysis for zero-shot generated-image detection; the geometric framing is novel and intellectually substantive.

**Importance of research question:** High. Zero-shot detection of generated images is timely and practically important as generative models proliferate.

**Claim support:** Moderate. The empirical claims are well-supported; the theoretical claims are partially supported, with a real gap in the theory-to-implementation bridge.

**Soundness of experiments:** Moderate-to-good. The benchmark is broad and comparisons are fair, but robustness coverage is shallow and computational cost is unreported.

**Clarity of writing:** Good. The paper is clearly structured and the mathematical exposition is accessible, though the significance of the CLIP remapping is understated.

**Value to the research community:** High. The empirical results substantially advance the zero-shot detection state of the art, and the theoretical framework opens new research directions even if not fully closed.

The paper makes a genuine and substantial empirical contribution (the first principled zero-shot diffusion-manifold detector, with large performance margins) alongside a theoretically motivated but imperfectly executed framework. The weaknesses—particularly the CLIP-space theory gap and the unvalidated approximation in Eq. (17)—are real and should be addressed, but they do not invalidate the core result. The limitations section is admirably honest. This clears the bar for acceptance with revisions.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>