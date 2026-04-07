## Summary

This paper introduces Cross-Modal Mechanistic Analysis (CMMA), a framework combining Multimodal Sparse Autoencoders (M-SAE) with causal intervention techniques to identify and manipulate interpretable cross-modal features in multimodal AIGC models. Through experiments on CLIP, LLaVA, and Stable Diffusion using 2.5M samples, the authors report discovering a consistent three-phase processing pattern (feature extraction, modal alignment, concept synthesis) and demonstrate that targeted feature interventions reduce hallucinations by 34.2% and improve semantic consistency by 28.7%.

## Strengths

- **Novel methodological contribution for multimodal interpretability:** The M-SAE architecture extends traditional sparse autoencoders with explicit cross-modal sparsity constraints (R_cross and R_align terms in Eq. 4-6), addressing the unique challenge of disentangling cross-modal features that prior unimodal SAE work could not handle. This is a principled extension that provides formal regularization for modality specialization.

- **Rigorous statistical framework:** The paper reports effect sizes (Cohen's d), 95% confidence intervals via bootstrap resampling, and applies multiple testing corrections (Benjamini-Hochberg, Holm-Bonferroni). This level of statistical rigor exceeds typical machine learning papers and provides meaningful uncertainty quantification around the claimed results.

- **Actionable intervention results:** Unlike purely observational interpretability work, the paper demonstrates practical applications—reducing hallucinations and improving semantic consistency through feature manipulation—suggesting immediate utility for model debugging and safety applications.

## Weaknesses

- **Inconsistency between abstract claims and experimental scope:** The abstract states experiments on "DALL-E variants" while Section 4.1.2 exclusively uses Stable Diffusion 2.1. DALL-E and Stable Diffusion have fundamentally different architectures (autoregressive vs. latent diffusion), making this a substantive discrepancy. The paper should either correct the abstract or explain why DALL-E was excluded.

- **Layer mapping is incoherent across architectures:** The paper claims a consistent three-phase pattern across all models with specific layer ranges (Phase 1: layers 1-4, Phase 2: layers 5-8, Phase 3: layers 9-12). However, LLaVA-7B is built on LLaMA-7B with 32 transformer layers, while CLIP ViT-B/32 has 12 vision transformer layers. The paper never explains how layer indices map across these different architectures, making the "consistent cross-architecture pattern" claim impossible to evaluate. The authors must either: (a) use relative layer positions (e.g., first 1/3, middle 1/3, final 1/3), or (b) explicitly state which layers were analyzed for each model.

- **No ablation study for regularization terms:** Equation 3 introduces three regularization terms (λ₁∥a∥₁, λ₂R_cross, λ₃R_align), but no experiments isolate their individual contributions. Without ablating R_cross and R_align separately, it is unclear whether the cross-modal regularization terms meaningfully improve over a standard SAE, or whether similar results could be achieved with standard sparsity alone.

- **Hallucination reduction methodology is underspecified:** Section 5.5.1 claims 34.2% hallucination reduction via "targeted interventions on hallucination-prone features" but does not explain how these features are identified. Is identification done in-sample or on held-out data? What mechanism determines which features to intervene on for a given input? Without this specification, the result is not reproducible.

- **Theoretical claims lack rigor:** Section 3.1.1 claims "the learned features satisfy modality specialization with probability at least 1−δ where δ decreases exponentially with the number of training samples" but provides only an informal proof sketch. The stated convergence property is asserted without bounding δ or specifying distributional assumptions. Either rigorous proof or removal of these claims is needed.

- **Reported effect sizes are unusually large:** The paper reports η² = 0.94–0.96 and Cohen's d > 2.0 across multiple experiments, values far exceeding conventional "large" effect thresholds (η² > 0.14, d > 0.8). While not impossible, such extraordinary effect sizes warrant scrutiny—particularly whether measurement methodology or data characteristics might inflate these values.

- **Compute budget raises reproducibility concerns:** Training SAEs with dictionary sizes up to 262K features on models as large as LLaVA-13B requires substantial compute. The paper states 8× A100-80GB GPUs were used but does not report training duration, FLOPs, or whether gradient checkpointing/other memory optimizations were required. This information is essential for reproducibility assessment.

## Nice-to-Haves

- **Comparison to alternative intervention methods:** Comparing hallucination reduction against fine-tuning or RLHF approaches would contextualize whether feature-level intervention is competitive with standard methods.

- **Reconstruction fidelity comparison:** Reporting reconstruction loss (MSE, SSIM) for M-SAE versus standard SAE would verify that cross-modal regularization does not degrade representation quality.

- **Clarification on CLIP's role:** CLIP is a discriminative contrastive model, not a generative model. While it underpins many AIGC systems, the paper should clarify whether interpretability insights from CLIP transfer to generative settings or remain specific to contrastive learning dynamics.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Duplicate section headings (5.3 and 5.4):** Formatting issue, not substantive criticism.

- **CLIP mischaracterization as AIGC model:** While technically CLIP is discriminative, the paper does not claim it generates content—it examines cross-modal representations. The abstract grouping is imprecise but does not undermine core claims.

- **Zero mixed features for CLIP ViT-B/32 (0.0 ± 0.0):** This is noted as suspicious but likely reflects the definition of "mixed" features rather than an error. The paper could clarify this, but it is not a fundamental flaw.

- **Statistical framework inconsistency (Benjamini-Hochberg vs. Holm-Bonferroni):** The two corrections apply to different analyses (CMIG significance vs. intervention analysis). While pre-specification would be ideal, using different corrections for different contexts is methodologically defensible.

## Novel Insights

The three-phase processing hypothesis—if validated with proper layer normalization—provides a compelling framework for understanding cross-modal integration that parallels cognitive science theories of human multimodal processing. The observation that cross-modal features are distributed across all layers rather than concentrated in a "fusion module" challenges the common architectural assumption that multimodal integration is localized, suggesting instead that sophisticated distributed processing underlies successful cross-modal understanding. This has implications for model design: capacity allocation should follow the 3:4:5 ratio across phases, though this specific ratio requires empirical validation.

## Suggestions

1. **Correct the layer mapping immediately:** Either use relative layer positions across architectures, or provide a table showing exactly which layer indices were analyzed for each model. The current absolute layer numbering cannot be correct for models with different depths.

2. **Add ablation experiments for regularization terms:** Train M-SAE with only λ₁, then add λ₂, then add λ₃, to isolate each term's contribution to interpretability and downstream intervention performance.

3. **Specify the hallucination-prone feature identification protocol:** Document the exact procedure (training data labels, held-out evaluation, or post-hoc analysis) to enable reproducibility.

4. **Correct the abstract:** Either add DALL-E experiments or remove the claim, and clarify the model scope.

5. **Report compute costs:** Include training time, memory usage, and any optimizations required for reproducibility.